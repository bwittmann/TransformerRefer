import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from plyfile import PlyData, PlyElement

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from utils.pc_utils import write_ply_rgb
from utils.box_util import get_3d_box, box3d_iou
from models.refnetV2 import RefNetV2
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from lib.config import CONF

# data
SCANNET_ROOT = CONF.PATH.SCANNET_SCANS  # TODO change if necessary
SCANNET_MESH = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean_2.ply")  # scene_id, scene_id
SCANNET_META = os.path.join(SCANNET_ROOT, "{}/{}.txt")  # scene_id, scene_id
SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DC = ScannetDatasetConfig()


def get_dataloader(args, scanrefer, all_scene_list, split):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_color=args.use_color, 
        use_height=(not args.no_height),
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return dataset, dataloader


def get_model(args):
    # create model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    detector_args = {
        'width': args.width,
        'bn_momentum': args.bn_momentum,
        'sync_bn': args.sync_bn,
        'dropout': args.dropout,
        'activation': args.activation,
        'nhead': args.nhead,
        'num_decoder_layers': args.num_decoder_layers,
        'dim_feedforward': args.dim_feedforward,
        'cross_position_embedding': args.cross_position_embedding,
        'size_cls_agnostic': args.size_cls_agnostic,
        'num_proposals': args.num_proposals,
        'sampling': args.sampling,
        'self_position_embedding': args.self_position_embedding,
        'num_features': args.num_features
    }
    model = RefNetV2(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference,
        detector_args=detector_args,
        emb_size=args.emb_size,
        use_multi_ref_gt=args.use_multi_ref_gt
    ).cuda()

    # load model
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, "model.pth")
    model.load_state_dict(torch.load(path), strict=False)

    # put model in eval
    model.eval()
    return model


def get_scanrefer(args):
    scanrefer = SCANREFER_TRAIN if args.use_train else SCANREFER_VAL
    all_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
    if args.scene_id:
        assert args.scene_id in all_scene_list, "The scene_id is not found"
        scene_list = [args.scene_id]
    else:
        scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
    scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]
    return scanrefer, scene_list


def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2], int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()


def write_bbox(bbox, mode, output_file):
    """
    bbox: (cx, cy, cz, lx, ly, lz, r), center and length in three axis, the last is the rotation
    output_file: string
    """
    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
        
        import math

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0] * vec3[0] + vec3[1] * vec3[1] + vec3[2] * vec3[2])
        
        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0, 0] = 1 + t * (x * x - 1)
            rot[0, 1] = z * s + t * x * y
            rot[0, 2] = -y * s + t * x * z
            rot[1, 0] = -z * s + t * x * y
            rot[1, 1] = 1 + t * (y * y - 1)
            rot[1, 2] = x * s + t * y * z
            rot[2, 0] = y * s + t * x * z
            rot[2, 1] = -x * s + t * y * z
            rot[2, 2] = 1 + t * (z * z - 1)
            return rot

        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks + 1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius*math.cos(theta), radius * math.sin(theta), height * i / stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append(np.array([(i + 1) * slices + i2, i * slices + i2, i * slices + i2p1], dtype=np.uint32))
                indices.append(np.array([(i + 1) * slices + i2, i * slices + i2p1, (i + 1) * slices + i2p1], dtype=np.uint32))
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if math.fabs(dotx) != 1.0:
                    axis = np.array([1, 0, 0]) - dotx * va
                else:
                    axis = np.array([0, 1, 0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3, 3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
            
        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    def get_bbox_corners(bbox):
        centers, lengths = bbox[:3], bbox[3:6]
        xmin, xmax = centers[0] - lengths[0] / 2, centers[0] + lengths[0] / 2
        ymin, ymax = centers[1] - lengths[1] / 2, centers[1] + lengths[1] / 2
        zmin, zmax = centers[2] - lengths[2] / 2, centers[2] + lengths[2] / 2
        corners = []
        corners.append(np.array([xmax, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymax, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmax]).reshape(1, 3))
        corners.append(np.array([xmax, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmin]).reshape(1, 3))
        corners.append(np.array([xmin, ymin, zmax]).reshape(1, 3))
        corners = np.concatenate(corners, axis=0)  # 8 x 3

        return corners

    radius = 0.03
    offset = [0, 0, 0]
    verts = []
    indices = []
    colors = []
    corners = get_bbox_corners(bbox)

    box_min = np.min(corners, axis=0)
    box_max = np.max(corners, axis=0)
    if args.color_bbox_pred == 'blue':
        palette = {
            0: [0, 255, 0], # gt
            1: [0, 0, 255]  # pred
        }
    elif args.color_bbox_pred == 'red':
        palette = {
            0: [0, 255, 0], # gt
            1: [255, 0, 0]  # pred
        }
    chosen_color = palette[mode]
    edges = get_bbox_edges(box_min, box_max)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c / 255 for c in chosen_color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)

    write_ply(verts, colors, indices, output_file)


def read_mesh(filename):
    """ read XYZ for each vertex."""
    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['red']
        vertices[:, 4] = plydata['vertex'].data['green']
        vertices[:, 5] = plydata['vertex'].data['blue']
    return vertices, plydata['face']


def export_mesh(vertices, faces):
    new_vertices = []
    for i in range(vertices.shape[0]):
        new_vertices.append(
            (
                vertices[i][0],
                vertices[i][1],
                vertices[i][2],
                vertices[i][3],
                vertices[i][4],
                vertices[i][5],
            )
        )
    vertices = np.array(
        new_vertices,
        dtype=[
            ("x", np.dtype("float32")), 
            ("y", np.dtype("float32")), 
            ("z", np.dtype("float32")),
            ("red", np.dtype("uint8")),
            ("green", np.dtype("uint8")),
            ("blue", np.dtype("uint8"))
        ]
    )
    vertices = PlyElement.describe(vertices, "vertex")
    return PlyData([vertices, faces])


def align_mesh(scene_id):
    vertices, faces = read_mesh(SCANNET_MESH.format(scene_id, scene_id))
    for line in open(SCANNET_META.format(scene_id, scene_id)).readlines():
        if 'axisAlignment' in line:
            axis_align_matrix = np.array([float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape((4, 4))
            break
    
    # align
    pts = np.ones((vertices.shape[0], 4))
    pts[:, :3] = vertices[:, :3]
    pts = np.dot(pts, axis_align_matrix.T)
    vertices[:, :3] = pts[:, :3]

    mesh = export_mesh(vertices, faces)
    return mesh


def dump_results(args, scanrefer, data, config, include_best_pred_box_ref):
    dump_dir = os.path.join(CONF.PATH.OUTPUT, args.folder, "vis")
    os.makedirs(dump_dir, exist_ok=True)

    # from inputs
    ids = data['scan_idx'].detach().cpu().numpy()
    point_clouds = data['point_clouds'].cpu().numpy()
    batch_size = point_clouds.shape[0]

    pcl_color = data["pcl_color"].detach().cpu().numpy()
    if args.use_color:
        pcl_color = (pcl_color * 256 + MEAN_COLOR_RGB).astype(np.int64)
    
    # from network outputs
    # detection
    pred_objectness = (data['objectness_scores'] > 0).squeeze(2).float().detach().cpu().numpy()
    pred_center = data['center'].detach().cpu().numpy()  # (B,K,3)
    pred_heading_class = torch.argmax(data['heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(data['heading_residuals'], 2, pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
    pred_size_class = torch.argmax(data['size_scores'], -1)  # B,num_proposal
    pred_size_residual = torch.gather(data['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 3))  # B,num_proposal,1,3
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3
    # reference
    pred_ref_scores = data["cluster_ref"].detach().cpu().numpy()
    if args.use_multi_ref_gt:
        pred_ref_scores_prob = F.sigmoid(data["cluster_ref"] * (data['objectness_scores'] > 0).squeeze(2).float() * data['pred_mask']).detach().cpu().numpy()
    else:
        pred_ref_scores_prob = F.softmax(data["cluster_ref"] * (data['objectness_scores'] > 0).squeeze(2).float() * data['pred_mask'], dim=1).detach().cpu().numpy()
    # post-processing
    nms_masks = data['pred_mask'].detach().cpu().numpy()  # B,num_proposal
    
    # ground truth
    gt_center = data['center_label'].cpu().numpy()  # (B,MAX_NUM_OBJ,3)
    gt_heading_class = data['heading_class_label'].cpu().numpy()  # B,K2
    gt_heading_residual = data['heading_residual_label'].cpu().numpy()  # B,K2
    gt_size_class = data['size_class_label'].cpu().numpy()  # B,K2
    gt_size_residual = data['size_residual_label'].cpu().numpy()  # B,K2,3
    # reference
    gt_ref_labels = data["ref_box_label"].detach().cpu().numpy()
    pred_gt_ref = data["cluster_labels"].detach().cpu().numpy()

    for i in range(batch_size):
        # basic info
        idx = ids[i]
        scene_id = scanrefer[idx]["scene_id"]
        object_id = scanrefer[idx]["object_id"]
        object_name = scanrefer[idx]["object_name"]
        ann_id = scanrefer[idx]["ann_id"]
    
        # scene_output
        scene_dump_dir = os.path.join(dump_dir, scene_id)
        if not os.path.exists(scene_dump_dir):
            os.mkdir(scene_dump_dir)

            # # Dump the original scene point clouds
            mesh = align_mesh(scene_id)
            mesh.write(os.path.join(scene_dump_dir, 'mesh.ply'))

            write_ply_rgb(point_clouds[i], pcl_color[i], os.path.join(scene_dump_dir, 'pc.ply'))

        # filter out the valid ground truth reference box
        assert gt_ref_labels[i].shape[0] == gt_center[i].shape[0]
        gt_ref_idx = np.argmax(gt_ref_labels[i], 0)

        # visualize the gt reference box
        # NOTE: for each object there should be only one gt reference box
        object_dump_dir = os.path.join(dump_dir, scene_id, "gt_{}_{}.ply".format(object_id, object_name))
        gt_obb = config.param2obb(gt_center[i, gt_ref_idx, 0:3], gt_heading_class[i, gt_ref_idx], gt_heading_residual[i, gt_ref_idx], gt_size_class[i, gt_ref_idx], gt_size_residual[i, gt_ref_idx])
        gt_bbox = get_3d_box(gt_obb[3:6], gt_obb[6], gt_obb[0:3])

        if not os.path.exists(object_dump_dir):
            write_bbox(gt_obb, 0, os.path.join(scene_dump_dir, 'gt_{}_{}.ply'.format(object_id, object_name)))
        
        # find the valid reference prediction
        pred_masks = nms_masks[i] * pred_objectness[i] == 1
        assert pred_ref_scores[i].shape[0] == pred_center[i].shape[0]
        pred_ref_idx = np.argmax(pred_ref_scores[i] * pred_masks, 0)

        # visualize the predicted reference box
        pred_obb = config.param2obb(pred_center[i, pred_ref_idx, 0:3], pred_heading_class[i, pred_ref_idx], pred_heading_residual[i, pred_ref_idx], pred_size_class[i, pred_ref_idx], pred_size_residual[i, pred_ref_idx])
        pred_bbox = get_3d_box(pred_obb[3:6], pred_obb[6], pred_obb[0:3])
        iou = box3d_iou(gt_bbox, pred_bbox)

        write_bbox(pred_obb, 1, os.path.join(scene_dump_dir, 'pred_{}_{}_{}_{:.5f}_{:.5f}.ply'.format(object_id, object_name, ann_id, pred_ref_scores_prob[i, pred_ref_idx], iou)))

        if not include_best_pred_box_ref:
            continue

        # visualize the predicted box that has the highest iou with gt ref box
        pred_gt_ref_idx = np.argmax(pred_gt_ref[i], 0)
        pred_gt_obb = config.param2obb(pred_center[i, pred_gt_ref_idx, 0:3], pred_heading_class[i, pred_gt_ref_idx], pred_heading_residual[i, pred_gt_ref_idx], pred_size_class[i, pred_gt_ref_idx], pred_size_residual[i, pred_gt_ref_idx])
        pred_gt_bbox = get_3d_box(pred_gt_obb[3:6], pred_gt_obb[6], pred_gt_obb[0:3])
        iou = box3d_iou(gt_bbox, pred_gt_bbox)

        write_bbox(pred_gt_obb, 1, os.path.join(scene_dump_dir, 'pred_gt_{}_{}_{}_iou_{:.5f}.ply'.format(object_id, object_name, ann_id, iou)))


def visualize(args):
    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "val")

    # model
    model = get_model(args)

    # config
    POST_DICT = {
        'remove_empty_box': True, 
        'use_3d_nms': True, 
        'nms_iou': 0.25,
        'use_old_type_nms': False, 
        'cls_nms': True, 
        'per_class_proposal': True,
        'conf_thresh': 0.05,
        'dataset_config': DC
    } if not args.no_nms else None
    
    # evaluate
    print("visualizing...")
    for data in tqdm(dataloader):
        for key in data:
            data[key] = data[key].cuda()

        # feed
        data = model(data)

        _, data = get_loss(
            end_points=data, 
            config=DC, 
            num_decoder_layers=args.num_decoder_layers,
            query_points_generator_loss_coef=args.query_points_generator_loss_coef,
            obj_loss_coef=args.obj_loss_coef,
            box_loss_coef=args.box_loss_coef,
            sem_cls_loss_coef=args.sem_cls_loss_coef,
            center_delta=args.center_delta,
            size_delta=args.size_delta,
            heading_delta=args.heading_delta,
            detection_loss_coef=args.detection_loss_coef,
            ref_loss_coef=args.ref_loss_coef,
            lang_loss_coef=args.lang_loss_coef,
            detection=True,
            reference=True,
            use_lang_classifier=(not args.no_lang_cls),
            use_votenet_objectness=args.use_votenet_objectness,
            use_multi_ref_gt=args.use_multi_ref_gt
        )

        data = get_eval(
            data_dict=data, 
            config=DC,
            reference=True, 
            use_lang_classifier=(not args.no_lang_cls),
            post_processing=POST_DICT,
            use_multi_ref_gt=args.use_multi_ref_gt
        )
        
        # visualize
        dump_results(args, scanrefer, data, DC, args.also_get_best_ref_pred_bbox)

    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model", required=True)
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--scene_id", type=str, help="scene id", default="")
    parser.add_argument("--batch_size", type=int, help="batch size", default=8)
    parser.add_argument('--num_points', type=int, default=50000, help='Point Number [default: 50000]')
    parser.add_argument('--num_scenes', type=int, default=-1, help='Number of scenes [default: -1]')
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--no_nms', action='store_true', help='do NOT use non-maximum suppression for post-processing.')
    parser.add_argument('--use_train', action='store_true', help='Use the training set.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_normal', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_multiview', action='store_true', help='Use multiview images.')
    parser.add_argument('--color_bbox_pred', type=str, default='red', help='Color of the bounding box of the predicted objects.')
    parser.add_argument("--use_bidir", action="store_true", help="use bi-directional GRU")
    parser.add_argument("--no_lang_cls", action="store_true", help="do NOT use language classifier")
    parser.add_argument("--no_reference", action="store_true", help="do NOT train the localization module")
    parser.add_argument("--emb_size", type=int, default=300, help="input size to GRU")

    # loss
    parser.add_argument('--use_votenet_objectness', action="store_true", help='use objectness as it is used by VoteNet')
    parser.add_argument('--use_multi_ref_gt', action="store_true", help='use multiple reference ground truths')
    parser.add_argument('--detection_loss_coef', default=1., type=float, help='loss weight for detection loss')
    parser.add_argument('--ref_loss_coef', default=0.1, type=float, help='loss weight for ref loss')
    parser.add_argument('--lang_loss_coef', default=0.1, type=float, help='loss weight for lang loss')
    parser.add_argument('--query_points_generator_loss_coef', default=0.8, type=float)
    parser.add_argument('--obj_loss_coef', default=0.1, type=float, help='loss weight for objectness loss')
    parser.add_argument('--box_loss_coef', default=1, type=float, help='loss weight for box loss')
    parser.add_argument('--sem_cls_loss_coef', default=0.1, type=float, help='loss weight for classification loss')
    parser.add_argument('--center_delta', default=1.0, type=float, help='delta for smoothl1 loss in center loss')
    parser.add_argument('--size_delta', default=1.0, type=float, help='delta for smoothl1 loss in size loss')
    parser.add_argument('--heading_delta', default=1.0, type=float, help='delta for smoothl1 loss in heading loss')

    # detector
    parser.add_argument("--num_proposals", type=int, default=256, help="proposal number")
    parser.add_argument("--width", type=int, default=1, help="PointNet backbone width ratio")
    parser.add_argument("--bn_momentum", type=float, default=0.1, help="batchnorm momentum")
    parser.add_argument("--sync_bn", action="store_true", help="converts all bn layers in SyncBatchNorm layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--activation", type=str, default='relu', choices=["relu", "gelu", "glu"], help="activation fct used in the decoder layers")
    parser.add_argument("--nhead", type=int, default=8, help="parallel attention heads in multihead attention")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="hidden size of the linear layers in the decoder")
    parser.add_argument("--cross_position_embedding", type=str, default='xyz_learned', choices=["none", "xyz_learned"], help="position embedding for cross-attention")
    parser.add_argument("--self_position_embedding", type=str, default='loc_learned', choices=["none", "xyz_learned", "loc_learned"], help="position embedding for self-attention")
    parser.add_argument("--size_cls_agnostic", action="store_true", help="use class agnostic predict heads")
    parser.add_argument("--sampling", type=str, default="kps", help="initial object candidate sampling")
    parser.add_argument("--num_features", type=int, default=288, help="number of features of the object proposals")
    parser.add_argument("--also_get_best_ref_pred_bbox", action="store_true", help="Will also output the bounding boxes of the predicted bbox that has the highest iou with the ground truth reference bbox (prefix 'pred_gt_').")

    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    visualize(args)
