import torch
import torch.nn as nn
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils.nn_distance import nn_distance
from lib.loss import smoothl1_loss, l1_loss, SigmoidFocalClassificationLoss, SoftmaxRankingLoss
from utils.box_util import get_3d_box_batch, box3d_iou_batch

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3


def compute_points_obj_cls_loss_hard_topk(end_points, topk):
    box_label_mask = end_points['box_label_mask']
    seed_inds = end_points['seed_inds'].long()  # B, K
    seed_xyz = end_points['seed_xyz']  # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, K
    gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
    gt_size = end_points['size_gts'][:, :, 0:3]  # B, K2, 3
    B = gt_center.shape[0]
    K = seed_xyz.shape[1]
    K2 = gt_center.shape[1]

    point_instance_label = end_points['point_instance_label']  # B, num_points
    object_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
    object_assignment_one_hot = torch.zeros((B, K, K2)).to(seed_xyz.device)
    object_assignment_one_hot.scatter_(2, object_assignment.unsqueeze(-1), 1)  # (B, K, K2)
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, K2, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, K2, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxK2
    euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + 100 * (1 - object_assignment_one_hot)  # BxKxK2
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxK2xK
    topk_inds = torch.topk(euclidean_dist1, topk, largest=False)[1] * box_label_mask[:, :, None] + (box_label_mask[:, :, None] - 1)  # BxK2xtopk
    topk_inds = topk_inds.long()  # BxK2xtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, K2xtopk
    batch_inds = torch.arange(B).unsqueeze(1).repeat(1, K2 * topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([batch_inds, topk_inds], -1).view(-1, 2).contiguous()

    objectness_label = torch.zeros((B, K + 1), dtype=torch.long).to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    objectness_label[objectness_label_mask < 0] = 0

    total_num_points = B * K
    end_points[f'points_hard_topk{topk}_pos_ratio'] = torch.sum(objectness_label.float()) / float(total_num_points)
    end_points[f'points_hard_topk{topk}_neg_ratio'] = 1 - end_points[f'points_hard_topk{topk}_pos_ratio']

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(seeds_obj_cls_logits.view(B, K, 1), objectness_label.unsqueeze(-1), weights=cls_weights)
    objectness_loss = cls_loss_src.sum() / B

    # Compute recall upper bound
    padding_array = torch.arange(0, B).to(point_instance_label.device) * 10000
    padding_array = padding_array.unsqueeze(1)  # B,1
    point_instance_label_mask = (point_instance_label < 0)  # B,num_points
    point_instance_label = point_instance_label + padding_array  # B,num_points
    point_instance_label[point_instance_label_mask] = -1
    num_gt_bboxes = torch.unique(point_instance_label).shape[0] - 1
    seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
    pos_points_instance_label = seed_instance_label * objectness_label + (objectness_label - 1)
    num_query_bboxes = torch.unique(pos_points_instance_label).shape[0] - 1
    if num_gt_bboxes > 0:
        end_points[f'points_hard_topk{topk}_upper_recall_ratio'] = num_query_bboxes / num_gt_bboxes
    return objectness_loss


def compute_objectness_loss_based_on_query_points(end_points, num_decoder_layers, votenet_objectness=False):
    """ Compute objectness loss for the proposals."""
    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    objectness_loss_sum = 0.0

    # Associate proposal and GT objects
    seed_inds = end_points['seed_inds'].long()  # B,num_seed in [0,num_points-1]
    gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
    query_points_sample_inds = end_points['query_points_sample_inds'].long()

    B = seed_inds.shape[0]
    K = query_points_sample_inds.shape[1]
    K2 = gt_center.shape[1]

    # get ground truths for each query point, for both objectness and instance label -> instance label / object
    # assignment is just for later (saved in end_points), not relevant for this loss
    seed_obj_gt = torch.gather(end_points['point_obj_mask'], 1, seed_inds)  # B,num_seed
    query_points_obj_gt = torch.gather(seed_obj_gt, 1, query_points_sample_inds)  # B,query_points

    seed_instance_label = torch.gather(end_points['point_instance_label'], 1, seed_inds)  # B,num_seed
    query_points_instance_label = torch.gather(seed_instance_label, 1, query_points_sample_inds)  # B,query_points

    # ground truth object assignment for the query points
    query_points_object_assignment = query_points_instance_label  # (B,K) with values in 0,1,...,K2-1
    query_points_object_assignment[query_points_object_assignment < 0] = K2 - 1  # set background points to the last gt bbox

    total_num_proposal = query_points_obj_gt.shape[0] * query_points_obj_gt.shape[1]

    for prefix in prefixes:
        if votenet_objectness:
            # Associate proposal and GT objects by point-to-point distances
            base_xyz = end_points[f'{prefix}base_xyz']
            dist1, ind1, dist2, _ = nn_distance(base_xyz, gt_center)  # dist1: BxK, dist2: BxK2
            object_assignment = ind1

            # Generate objectness label and mask
            # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
            # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
            euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
            objectness_label = torch.zeros((B, K), dtype=torch.long).cuda()
            objectness_mask = torch.zeros((B, K)).cuda()
            objectness_label[euclidean_dist1 < NEAR_THRESHOLD] = 1
            objectness_mask[euclidean_dist1 < NEAR_THRESHOLD] = 1
            objectness_mask[euclidean_dist1 > FAR_THRESHOLD] = 1

            end_points[f'{prefix}objectness_mask'] = objectness_mask
            end_points[f'{prefix}objectness_label'] = objectness_label
            end_points[f'{prefix}object_assignment'] = object_assignment
        else:
            # Only ones means we want to have all objectness predictions in the loss
            objectness_mask = torch.ones((B, K)).cuda()

            end_points[f'{prefix}objectness_mask'] = objectness_mask
            end_points[f'{prefix}objectness_label'] = query_points_obj_gt
            end_points[f'{prefix}object_assignment'] = query_points_object_assignment

        end_points[f'{prefix}pos_ratio'] = torch.sum(end_points[f'{prefix}objectness_label'].float().cuda()) / float(total_num_proposal)
        end_points[f'{prefix}neg_ratio'] = torch.sum(objectness_mask.float()) / float(total_num_proposal) - end_points[f'{prefix}pos_ratio']

        # Compute objectness loss
        objectness_scores = end_points[f'{prefix}objectness_scores']
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = objectness_mask.clone()
        cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        cls_weights /= torch.clamp(cls_normalizer, min=1.0)
        cls_loss_src = criterion(objectness_scores.transpose(2, 1).contiguous().view(B, K, 1), query_points_obj_gt.unsqueeze(-1), weights=cls_weights)
        objectness_loss = cls_loss_src.sum() / B

        end_points[f'{prefix}objectness_loss'] = objectness_loss
        objectness_loss_sum += objectness_loss

    return objectness_loss_sum, end_points


def compute_box_and_sem_cls_loss(end_points, config, num_decoder_layers,
                                 center_loss_type='smoothl1', center_delta=1.0,
                                 size_loss_type='smoothl1', size_delta=1.0,
                                 heading_loss_type='smoothl1', heading_delta=1.0,
                                 size_cls_agnostic=False):
    """ Compute 3D bounding box and semantic classification loss."""
    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    mean_size_arr = config.mean_size_arr
    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0
    for prefix in prefixes:
        object_assignment = end_points[f'{prefix}object_assignment']
        batch_size = object_assignment.shape[0]

        # Compute center loss
        pred_center = end_points[f'{prefix}center']
        gt_center = end_points['center_label'][:, :, 0:3]

        if center_loss_type == 'smoothl1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = smoothl1_loss(assigned_gt_center - pred_center, delta=center_delta)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        elif center_loss_type == 'l1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = l1_loss(assigned_gt_center - pred_center)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute heading loss
        heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
        heading_class_loss = criterion_heading_class(end_points[f'{prefix}heading_scores'].transpose(2, 1), heading_class_label)  # (B,K)
        heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)
        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
        heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
        heading_residual_normalized_error = torch.sum(end_points[f'{prefix}heading_residuals_normalized'] * heading_label_one_hot, -1) - heading_residual_normalized_label

        if heading_loss_type == 'smoothl1':
            heading_residual_normalized_loss = heading_delta * smoothl1_loss(heading_residual_normalized_error, delta=heading_delta)  # (B,K)
            heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        elif heading_loss_type == 'l1':
            heading_residual_normalized_loss = l1_loss(heading_residual_normalized_error)  # (B,K)
            heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute size loss
        if size_cls_agnostic:
            pred_size = end_points[f'{prefix}pred_size']
            size_label = torch.gather(
                end_points['size_gts'], 1,
                object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
            size_error = pred_size - size_label
            if size_loss_type == 'smoothl1':
                size_loss = size_delta * smoothl1_loss(size_error, delta=size_delta)  # (B,K,3) -> (B,K)
                size_loss = torch.sum(size_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
            elif size_loss_type == 'l1':
                size_loss = l1_loss(size_error)  # (B,K,3) -> (B,K)
                size_loss = torch.sum(size_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
            else:
                raise NotImplementedError
        else:
            size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment)  # select (B,K) from (B,K2)
            criterion_size_class = nn.CrossEntropyLoss(reduction='none')
            size_class_loss = criterion_size_class(end_points[f'{prefix}size_scores'].transpose(2, 1), size_class_label)  # (B,K)
            size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
            size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)
            size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
            size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
            size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3).contiguous()  # (B,K,num_size_cluster,3)
            predicted_size_residual_normalized = torch.sum(end_points[f'{prefix}size_residuals_normalized'].contiguous() * size_label_one_hot_tiled, 2)  # (B,K,3)
            mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0)  # (1,1,num_size_cluster,3)
            mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
            size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)
            size_residual_normalized_error = predicted_size_residual_normalized - size_residual_label_normalized

            if size_loss_type == 'smoothl1':
                size_residual_normalized_loss = size_delta * smoothl1_loss(size_residual_normalized_error, delta=size_delta)  # (B,K,3) -> (B,K)
                size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
            elif size_loss_type == 'l1':
                size_residual_normalized_loss = l1_loss(size_residual_normalized_error)  # (B,K,3) -> (B,K)
                size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
            else:
                raise NotImplementedError

        # Semantic cls loss
        sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        sem_cls_loss = criterion_sem_cls(end_points[f'{prefix}sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        # store
        end_points[f'{prefix}center_loss'] = center_loss
        end_points[f'{prefix}heading_cls_loss'] = heading_class_loss
        end_points[f'{prefix}heading_reg_loss'] = heading_residual_normalized_loss
        if size_cls_agnostic:
            end_points[f'{prefix}size_reg_loss'] = size_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + size_loss
        else:
            end_points[f'{prefix}size_cls_loss'] = size_class_loss
            end_points[f'{prefix}size_reg_loss'] = size_residual_normalized_loss
            box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + 0.1 * size_class_loss + size_residual_normalized_loss
        end_points[f'{prefix}box_loss'] = box_loss
        end_points[f'{prefix}sem_cls_loss'] = sem_cls_loss

        box_loss_sum += box_loss
        sem_cls_loss_sum += sem_cls_loss

    return box_loss_sum, sem_cls_loss_sum, end_points


def compute_reference_loss(data_dict, config, use_multi_ref_gt=False):
    """Compute cluster reference loss"""
    MULTI_REF_IOU_THRESHOLD = 0.3

    # unpack
    cluster_preds = data_dict["cluster_ref"]  # (B, num_proposal)

    # predicted bbox
    pred_ref = data_dict['cluster_ref'].detach().cpu().numpy()  # (B,)
    pred_center = data_dict['center'].detach().cpu().numpy()  # (B,K,3)
    pred_heading_class = torch.argmax(data_dict['heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
    pred_size_class = torch.argmax(data_dict['size_scores'], -1)  # B,num_proposal
    pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3))  # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy()
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3

    # ground truth bbox
    gt_center = data_dict['ref_center_label'].cpu().numpy()  # (B,3)
    gt_heading_class = data_dict['ref_heading_class_label'].cpu().numpy()  # B
    gt_heading_residual = data_dict['ref_heading_residual_label'].cpu().numpy()  # B
    gt_size_class = data_dict['ref_size_class_label'].cpu().numpy()  # B
    gt_size_residual = data_dict['ref_size_residual_label'].cpu().numpy()  # B,3
    # convert gt bbox parameters to bbox corners
    gt_obb_batch = config.param2obb_batch(gt_center[:, 0:3], gt_heading_class, gt_heading_residual, gt_size_class, gt_size_residual)
    gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])

    # compute the iou score for all predicted positive ref
    batch_size, num_proposals = cluster_preds.shape
    labels = np.zeros((batch_size, num_proposals))
    weights = np.zeros((batch_size, num_proposals))
    for i in range(pred_ref.shape[0]):
        # convert the bbox parameters to bbox corners
        pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i], pred_heading_residual[i], pred_size_class[i], pred_size_residual[i])
        pred_bbox_batch = get_3d_box_batch(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
        ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[i], (num_proposals, 1, 1)))
        max_idx = ious.argmax()

        if use_multi_ref_gt:
            # set all predicted bboxes with iou over threshold as gt for the ref box
            labels[i, ious > MULTI_REF_IOU_THRESHOLD] = 1
            num_gt = labels[i, :].sum()
            # weigh the ref scores according to their ious and scale by inbalance,
            # the negatives -> if close to iou threshold almost 0 weight, if 0 iou 1 weight
            neg_scale = (num_proposals / (num_proposals - num_gt + 1e-8))
            pos_scale = (num_proposals / (num_gt + 1e-8))
            weights[i, ious < MULTI_REF_IOU_THRESHOLD] = neg_scale * (1 - (ious[ious < MULTI_REF_IOU_THRESHOLD] / MULTI_REF_IOU_THRESHOLD))
            weights[i, ious > MULTI_REF_IOU_THRESHOLD] = pos_scale * (1 / ious[max_idx] + 1e-8) * ious[ious > MULTI_REF_IOU_THRESHOLD]
        else:
            # treat the bbox with highest iou score as the gt
            labels[i, max_idx] = 1

    cluster_labels = torch.FloatTensor(labels).cuda()

    # reference loss
    criterion = SoftmaxRankingLoss()
    if use_multi_ref_gt:
        # multiply by X to get comparable magnitude
        loss_weights = torch.FloatTensor(5 * weights).cuda()
        criterion = nn.BCEWithLogitsLoss(weight=loss_weights)
    loss = criterion(cluster_preds, cluster_labels.float().clone())

    return loss, cluster_preds, cluster_labels


def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(data_dict["lang_scores"], data_dict["object_cat"])
    return loss


def get_loss(end_points,
             config,
             num_decoder_layers=6,
             query_points_generator_loss_coef=0.8, obj_loss_coef=0.1, box_loss_coef=1., sem_cls_loss_coef=0.1,
             detection_loss_coef=1.0, ref_loss_coef=0.1, lang_loss_coef=0.1,
             query_points_obj_topk=5,
             center_loss_type='smoothl1', center_delta=1.0,
             size_loss_type='smoothl1', size_delta=1.0,
             heading_loss_type='smoothl1', heading_delta=1.0,
             size_cls_agnostic=False,
             detection=True,
             reference=True,
             use_lang_classifier=False,
             use_votenet_objectness=False,
             use_multi_ref_gt=False):
    # detection losses:
    # sampling loss
    query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(end_points, query_points_obj_topk)

    # Obj loss
    objectness_loss_sum, end_points = compute_objectness_loss_based_on_query_points(end_points, num_decoder_layers, votenet_objectness=use_votenet_objectness)

    # Box loss and sem cls loss
    box_loss_sum, sem_cls_loss_sum, end_points = compute_box_and_sem_cls_loss(
        end_points, config, num_decoder_layers,
        center_loss_type, center_delta=center_delta,
        size_loss_type=size_loss_type, size_delta=size_delta,
        heading_loss_type=heading_loss_type, heading_delta=heading_delta,
        size_cls_agnostic=size_cls_agnostic
    )

    # store - we store the loss af the last decoder layer to be logged later
    if detection:
        end_points['query_points_generation_loss'] = query_points_generation_loss
        end_points['objectness_loss'] = end_points['last_objectness_loss']
        end_points['sum_heads_objectness_loss'] = objectness_loss_sum
        end_points['center_loss'] = end_points['last_center_loss']
        end_points['heading_cls_loss'] = end_points['last_heading_cls_loss']
        end_points['heading_reg_loss'] = end_points['last_heading_reg_loss']
        end_points['size_cls_loss'] = end_points['last_size_cls_loss']
        end_points['size_reg_loss'] = end_points['last_size_reg_loss']
        end_points['sem_cls_loss'] = end_points['last_sem_cls_loss']
        end_points['sum_heads_sem_cls_loss'] = sem_cls_loss_sum
        end_points['box_loss'] = end_points['last_box_loss']
        end_points['sum_heads_box_loss'] = box_loss_sum
    else:
        end_points['query_points_generation_loss'] = torch.zeros(1)[0].cuda()
        end_points['objectness_loss'] = torch.zeros(1)[0].cuda()
        end_points['sum_heads_objectness_loss'] = torch.zeros(1)[0].cuda()
        end_points['center_loss'] = torch.zeros(1)[0].cuda()
        end_points['heading_cls_loss'] = torch.zeros(1)[0].cuda()
        end_points['heading_reg_loss'] = torch.zeros(1)[0].cuda()
        end_points['size_cls_loss'] = torch.zeros(1)[0].cuda()
        end_points['size_reg_loss'] = torch.zeros(1)[0].cuda()
        end_points['sem_cls_loss'] = torch.zeros(1)[0].cuda()
        end_points['sum_heads_sem_cls_loss'] = torch.zeros(1)[0].cuda()
        end_points['box_loss'] = torch.zeros(1)[0].cuda()
        end_points['sum_heads_box_loss'] = torch.zeros(1)[0].cuda()

    # reference loss:
    if reference:
        ref_loss, _, cluster_labels = compute_reference_loss(end_points, config, use_multi_ref_gt)
        end_points["cluster_labels"] = cluster_labels
        end_points["ref_loss"] = ref_loss
    else:
        end_points["cluster_labels"] = end_points['last_objectness_label'].new_zeros(end_points['last_objectness_label'].shape).cuda()
        end_points["cluster_ref"] = end_points['last_objectness_label'].new_zeros(end_points['last_objectness_label'].shape).float().cuda()
        end_points["ref_loss"] = torch.zeros(1)[0].cuda()

    # language cls loss
    if reference and use_lang_classifier:
        end_points["lang_loss"] = compute_lang_classification_loss(end_points)
    else:
        end_points["lang_loss"] = torch.zeros(1)[0].cuda()

    # get total loss
    detection_loss = query_points_generator_loss_coef * end_points['query_points_generation_loss'] + \
                     (1.0 / (num_decoder_layers + 1)) * (obj_loss_coef * end_points['sum_heads_objectness_loss'] +
                                                         box_loss_coef * end_points['sum_heads_box_loss'] +
                                                         sem_cls_loss_coef * end_points['sum_heads_sem_cls_loss'])

    loss = detection_loss_coef * detection_loss + ref_loss_coef * end_points["ref_loss"] + lang_loss_coef * end_points["lang_loss"]
    loss *= 10
    end_points['loss'] = loss

    # Rename scores and residuals from last layer to general
    end_points['objectness_label'] = end_points['last_objectness_label']
    end_points['objectness_mask'] = end_points['last_objectness_mask']
    end_points['object_assignment'] = end_points['last_object_assignment']
    end_points['pos_ratio'] = end_points['last_pos_ratio']
    end_points['neg_ratio'] = end_points['last_neg_ratio']

    return loss, end_points
