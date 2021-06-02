import os
import sys
import json
import argparse
import torch
import numpy as np
import time

from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy


sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from lib.config import CONF
from lib.dataset import ScannetReferenceDataset
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_loss
from lib.loss_helper_detector import get_loss_detector
from models.refnet import RefNet
from models.refnetV2 import RefNetV2
from utils.logger_for_det_eval import setup_logger
from data.scannet.model_util_scannet import ScannetDatasetConfig

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

DATASET_CONFIG = ScannetDatasetConfig()


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
    print("evaluate on {} samples".format(len(dataset)))

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataset, dataloader


def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer():
    scene_list = get_scannet_scene_list("val")
    scanrefer = []
    for scene_id in scene_list:
        data = deepcopy(SCANREFER_TRAIN[0])
        data["scene_id"] = scene_id
        scanrefer.append(data)
    return scanrefer, scene_list


def get_model(args, config):
    # load model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(
        not args.no_height)

    if args.transformer:
        model = RefNetV2(
            num_class=config.num_class,
            num_heading_bin=config.num_heading_bin,
            num_size_cluster=config.num_size_cluster,
            mean_size_arr=config.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            use_lang_classifier=(not args.no_lang_cls),
            use_bidir=args.use_bidir
        ).cuda()
    else:
        model = RefNet(
            num_class=config.num_class,
            num_heading_bin=config.num_heading_bin,
            num_size_cluster=config.num_size_cluster,
            mean_size_arr=config.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            use_lang_classifier=(not args.no_lang_cls),
            use_bidir=args.use_bidir
        ).cuda()

    model_name = "model.pth"
    path = os.path.join(CONF.PATH.OUTPUT, args.folder, model_name)
    model.load_state_dict(torch.load(path), strict=False)

    model.eval()
    return model


def evaluate_one_time(test_loader, DATASET_CONFIG, CONFIG_DICT, AP_IOU_THRESHOLDS, model, args, repeat=0):
    prefixes = ['ScanRefer model only object detection, ']

    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
                          for iou_thresh in AP_IOU_THRESHOLDS]
    mAPs = [[iou_thresh, {k: 0 for k in prefixes}] for iou_thresh in AP_IOU_THRESHOLDS]

    model.eval()  # set model to eval mode (for bn and dp)

    batch_pred_map_cls_dict = {k: [] for k in prefixes}
    batch_gt_map_cls_dict = {k: [] for k in prefixes}

    for batch_data_label in tqdm(test_loader):
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        necessary_data_keys = ["point_clouds", "lang_feat", "lang_len"]
        inputs = {key: value for key, value in batch_data_label.items() if key in necessary_data_keys}
        with torch.no_grad():
            end_points = model(inputs)

        # Compute loss
        for key in batch_data_label:
            if key not in necessary_data_keys:
                assert (key not in end_points)
            end_points[key] = batch_data_label[key]
        if args.transformer:
            _, end_points = get_loss_detector(
                end_points=end_points,
                config=DATASET_CONFIG,
                num_decoder_layers=6,
                detection=True,
                reference=False,
                use_lang_classifier=not args.no_lang_cls,
                use_votenet_objectness=args.use_votenet_objectness
            )
        else:
            _, end_points = get_loss(
                data_dict=end_points,
                config=DATASET_CONFIG,
                detection=True,
                reference=False
            )

        for prefix in prefixes:
            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT, args.transformer)
            batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)
            batch_pred_map_cls_dict[prefix].append(batch_pred_map_cls)
            batch_gt_map_cls_dict[prefix].append(batch_gt_map_cls)

    for prefix in prefixes:
        for (batch_pred_map_cls, batch_gt_map_cls) in zip(batch_pred_map_cls_dict[prefix],
                                                          batch_gt_map_cls_dict[prefix]):
            for ap_calculator in ap_calculator_list:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
        # Evaluate average precision
        for i, ap_calculator in enumerate(ap_calculator_list):
            metrics_dict = ap_calculator.compute_metrics()
            logger.info(f'===================> (it. {repeat + 1}) {prefix} IOU THRESH: {AP_IOU_THRESHOLDS[i]} <==================')
            for key in metrics_dict:
                logger.info(f'{key} {metrics_dict[key]}')

            mAPs[i][1][prefix] = metrics_dict['mAP']
            ap_calculator.reset()
    for mAP in mAPs:
        logger.info(f'T[{time}] IoU[{mAP[0]}]: ' +
                    ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))
    return mAPs


def eval(args, avg_times=5):
    print("evaluate detection...")
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer()

    _, test_loader = get_dataloader(args, scanrefer, scene_list, "val")
    n_data = len(test_loader.dataset)
    logger.info(f"length of testing dataset: {n_data}")

    model = get_model(args, DATASET_CONFIG)
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        logger.info("Let's use %d GPUs!" % (torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box': (not args.faster_eval), 'use_3d_nms': True, 'nms_iou': args.nms_iou,
                   'use_old_type_nms': args.use_old_type_nms, 'cls_nms': True,
                   'per_class_proposal': True,
                   'conf_thresh': args.conf_thresh, 'dataset_config': DATASET_CONFIG}

    logger.info(str(datetime.now()))
    mAPs_times = [None for i in range(avg_times)]
    for i in range(avg_times):
        np.random.seed(i + args.rng_seed)
        mAPs = evaluate_one_time(test_loader, DATASET_CONFIG, CONFIG_DICT, args.ap_iou_thresholds,
                                 model, args, i)
        mAPs_times[i] = mAPs

    mAPs_avg = mAPs.copy()

    for i, mAP in enumerate(mAPs_avg):
        for key in mAP[1].keys():
            avg = 0
            for t in range(avg_times):
                cur = mAPs_times[t][i][1][key]
                avg += cur
            avg /= avg_times
            mAP[1][key] = avg

    for mAP in mAPs_avg:
        logger.info(f'AVG IoU[{mAP[0]}]: \n' +
                    ''.join([f'{key}: {mAP[1][key]:.4f} \n' for key in sorted(mAP[1].keys())]))

    for mAP in mAPs_avg:
        logger.info(f'AVG IoU[{mAP[0]}]: \t' +
                    ''.join([f'{key}: {mAP[1][key]:.4f} \t' for key in sorted(mAP[1].keys())]))


def parse_option():
    parser = argparse.ArgumentParser()
    # Eval
    parser.add_argument('--avg_times', default=5, type=int, help='Average times')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument('--dump_dir', default='dump', help='Dump dir to save sample outputs [default: None]')
    parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
    parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
    parser.add_argument('--conf_thresh', type=float, default=0.0,
                        help='Filter out predictions with obj prob less than it. [default: 0.05]')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5], nargs='+',
                        help='A list of AP IoU thresholds [default: 0.25,0.5]')
    parser.add_argument('--faster_eval', action='store_true',
                        help='Faster evaluation by skippling empty bounding box removal.')

    # Transformer
    parser.add_argument('--nhead', default=8, type=int, help='multi-head number')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers')

    # Loss
    parser.add_argument('--query_points_generator_loss_coef', default=0.8, type=float)
    parser.add_argument('--obj_loss_coef', default=0.1, type=float, help='Loss weight for objectness loss')
    parser.add_argument('--box_loss_coef', default=1, type=float, help='Loss weight for box loss')
    parser.add_argument('--sem_cls_loss_coef', default=0.1, type=float, help='Loss weight for classification loss')
    parser.add_argument('--center_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--center_delta', default=1.0, type=float, help='delta for smoothl1 loss in center loss')
    parser.add_argument('--size_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--size_delta', default=1.0, type=float, help='delta for smoothl1 loss in size loss')
    parser.add_argument('--heading_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--heading_delta', default=1.0, type=float, help='delta for smoothl1 loss in heading loss')
    parser.add_argument('--query_points_obj_topk', default=4, type=int, help='query_points_obj_topk')
    parser.add_argument('--size_cls_agnostic', action='store_true', help='Use class-agnostic size prediction.')

    # Data
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 8]')
    parser.add_argument('--dataset', default='scannet', help='Dataset name. sunrgbd or scannet. [default: scannet]')
    parser.add_argument('--num_points', type=int, default=50000, help='Point Number [default: 50000]')

    # from scanrefer
    parser.add_argument("--folder", type=str, help="Folder containing the model")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--transformer", action="store_true", help="Use the transformer for object detection")
    parser.add_argument("--use_votenet_objectness", action="store_true",
                        help="Use VoteNet's objectness labeling with transformer object detection")

    args, unparsed = parser.parse_known_args()

    return args


if __name__ == '__main__':
    opt = parse_option()

    opt.dump_dir = os.path.join(opt.dump_dir, f'eval_{opt.dataset}_{int(time.time())}_{np.random.randint(100000000)}')
    logger = setup_logger(output=opt.dump_dir, name="eval")

    eval(opt, opt.avg_times)
