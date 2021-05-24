import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from datetime import datetime
from copy import deepcopy
from collections import OrderedDict

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.pointnet2.pytorch_utils import BNMomentumScheduler
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.config import CONF
from models.refnet import RefNet
from models.refnetV2 import RefNetV2
from models.detector import GroupFreeDetector
from utils.lr_scheduler import get_scheduler

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()

# TODO: for debugging warnings
#np.seterr(all='raise')


def get_dataloader(args, scanrefer, all_scene_list, split, config, augment):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer[split],
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        augment = augment
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return dataset, dataloader


def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(
        not args.no_height)

    if args.transformer:
        model = RefNetV2(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            use_lang_classifier=(not args.no_lang_cls),
            use_bidir=args.use_bidir
        )

        # pretrained transformer directly from GroupFreeDetector weights
        if args.use_pretrained_transformer:
            # load model
            print("loading pretrained GroupFreeDetector...")
            pretrained_detector = GroupFreeDetector(num_class=DC.num_class,
                                                    num_heading_bin=DC.num_heading_bin,
                                                    num_size_cluster=DC.num_size_cluster,
                                                    mean_size_arr=DC.mean_size_arr,
                                                    input_feature_dim=input_channels,
                                                    num_proposal=args.num_proposals,
                                                    self_position_embedding='loc_learned')

            # model created with nn.DataParallel -> need to create new ordered dict and remove "module" prefix
            checkpoint = torch.load(args.use_pretrained_transformer, map_location='cpu')  # map_location='cpu'
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            pretrained_detector.load_state_dict(new_state_dict)
            del checkpoint
            torch.cuda.empty_cache()

            model.detector = pretrained_detector

        # from pretrained scanrefer model with transformer
        elif args.use_pretrained:
            # load model
            print("loading pretrained ScanRefer transformer detection...")
            pretrained_model = RefNetV2(
                num_class=DC.num_class,
                num_heading_bin=DC.num_heading_bin,
                num_size_cluster=DC.num_size_cluster,
                mean_size_arr=DC.mean_size_arr,
                num_proposal=args.num_proposals,
                input_feature_dim=input_channels,
                use_lang_classifier=(not args.no_lang_cls),
                use_bidir=args.use_bidir
            )

            pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
            pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

            # mount
            model.detector = pretrained_model.detector

        if args.no_detection:
            # freeze detector
            for param in model.detector.parameters():
                param.requires_grad = False

    else:
        model = RefNet(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            input_feature_dim=input_channels,
            num_proposal=args.num_proposals,
            use_lang_classifier=(not args.no_lang_cls),
            use_bidir=args.use_bidir,
            no_reference=args.no_reference
        )

        # trainable model
        if args.use_pretrained:
            # load model
            print("loading pretrained VoteNet...")
            pretrained_model = RefNet(
                num_class=DC.num_class,
                num_heading_bin=DC.num_heading_bin,
                num_size_cluster=DC.num_size_cluster,
                mean_size_arr=DC.mean_size_arr,
                num_proposal=args.num_proposals,
                input_feature_dim=input_channels,
                use_bidir=args.use_bidir,
                no_reference=True
            )

            pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
            pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

            # mount
            model.backbone_net = pretrained_model.backbone_net
            model.vgen = pretrained_model.vgen
            model.proposal = pretrained_model.proposal

            if args.no_detection:
                # freeze pointnet++ backbone
                for param in model.backbone_net.parameters():
                    param.requires_grad = False

                # freeze voting
                for param in model.vgen.parameters():
                    param.requires_grad = False

                # freeze detector
                for param in model.proposal.parameters():
                    param.requires_grad = False

    # to CUDA
    model = model.cuda()

    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def get_solver(args, dataloader):
    # get model
    model = get_model(args)

    # get optimizer
    if args.transformer:

        non_detector_params = []
        detector_params = []
        detector_decoder_params = []

        for param_name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "detector" in param_name:
                if "decoder" in param_name:
                    detector_decoder_params.append(param)
                else:
                    detector_params.append(param)
            else:
                non_detector_params.append(param)

        param_dicts = [
            {"params": non_detector_params},
            {"params": detector_params,
             "lr": args.t_learning_rate,
             "weight_decay": args.t_weight_decay},
            {"params": detector_decoder_params,
             "lr": args.t_decoder_learning_rate,
             "weight_decay": args.t_weight_decay}
        ]
        optimizer = optim.AdamW(param_dicts,
                                lr=args.lr,
                                weight_decay=args.wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # get scheduler
    if args.transformer:
        lr_scheduler = get_scheduler(optimizer, args.epoch, args.t_lr_scheduler,
                                     args.t_lr_decay_epochs, args.t_lr_decay_rate, args.t_warmup_epoch,
                                     args.t_warmup_multiplier)
        bn_scheduler = None
    else:
        # scheduler parameters for training solely the detection pipeline
        lr_decay_step = [80, 120, 160] if args.no_reference else None
        lr_decay_rate = 0.1 if args.no_reference else None
        bn_decay_step = 20 if args.no_reference else None
        bn_decay_rate = 0.5 if args.no_reference else None

        # lr scheduler
        if lr_decay_step and lr_decay_rate:
            if isinstance(lr_decay_step, list):
                lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
            else:
                lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
        else:
            lr_scheduler = None

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            it = -1
            start_epoch = 0
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * bn_decay_rate ** (int(it / bn_decay_step)), BN_MOMENTUM_MAX)
            bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)
        else:
            bn_scheduler = None

    # load checkpoint
    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_" + args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    trans_args = {"query_points_generator_loss_coef": args.t_query_points_generator_loss_coef,
                  "obj_loss_coef": args.t_obj_loss_coef,
                  "box_loss_coef": args.t_box_loss_coef,
                  "sem_cls_loss_coef": args.t_sem_cls_loss_coef,
                  "center_delta": args.t_center_delta,
                  "size_delta": args.t_size_delta,
                  "heading_delta": args.t_heading_delta,
                  "detection_loss_coef": args.t_detection_loss_coef,
                  "ref_loss_coef": args.t_ref_loss_coef,
                  "lang_loss_coef": args.t_lang_loss_coef}

    # get solver
    solver = Solver(
        model=model,
        config=DC,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        bn_scheduler=bn_scheduler,
        clip_norm=args.t_clip_norm,
        stamp=stamp,
        no_validation=args.no_validation,
        val_step=args.val_step,
        detection=not args.no_detection,
        reference=not args.no_reference,
        use_lang_classifier=not args.no_lang_cls,
        use_trans=args.transformer,
        trans_args=trans_args
    )
    num_params = get_num_params(model)

    return solver, num_params, root


def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes):
    if args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        if num_scenes == -1:
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes

        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]

        # filter data in chosen scenes
        new_scanrefer_train = []
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)

        new_scanrefer_val = scanrefer_val

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list


def train(args):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }

    # dataloader
    if args.augment:
        train_dataset, train_dataloader = get_dataloader(args, scanrefer, all_scene_list, "train", DC, True)
    else:
        train_dataset, train_dataloader = get_dataloader(args, scanrefer, all_scene_list, "train", DC, False)

    val_dataset, val_dataloader = get_dataloader(args, scanrefer, all_scene_list, "val", DC, False)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--num_workers", type=int, help="number of workers for dataloader", default=4)
    parser.add_argument("--batch_size", type=int, help="batch size", default=14)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--num_points", type=int, default=50000, help="Point Number [default: 50000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--augment", action="store_true", help="Use data augmentation.") 
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_pretrained", type=str, help="Specify the folder name in outputs containing the "
                                                           "pretrained model.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--no_validation", action="store_true", help="Do NOT validate. Only for development debugging.")

    # transformer specific options
    parser.add_argument("--transformer", action="store_true", help="Use the transformer for object detection")
    parser.add_argument("--use_pretrained_transformer", type=str, help="Specify the absolute file path for pretrained "
                                                                       "GroupFreeDetector module.")

    parser.add_argument('--t_detection_loss_coef', default=1., type=float, help='Loss weight for detection loss')
    parser.add_argument('--t_ref_loss_coef', default=0.1, type=float, help='Loss weight for ref loss')
    parser.add_argument('--t_lang_loss_coef', default=0.1, type=float, help='Loss weight for lang loss')

    parser.add_argument('--t_query_points_generator_loss_coef', default=0.8, type=float)
    parser.add_argument('--t_obj_loss_coef', default=0.1, type=float, help='Loss weight for objectness loss')
    parser.add_argument('--t_box_loss_coef', default=1, type=float, help='Loss weight for box loss')
    parser.add_argument('--t_sem_cls_loss_coef', default=0.1, type=float, help='Loss weight for classification loss')
    parser.add_argument('--t_center_delta', default=1.0, type=float, help='delta for smoothl1 loss in center loss')
    parser.add_argument('--t_size_delta', default=1.0, type=float, help='delta for smoothl1 loss in size loss')
    parser.add_argument('--t_heading_delta', default=1.0, type=float, help='delta for smoothl1 loss in heading loss')

    parser.add_argument('--t_weight_decay', type=float, default=0.0005,
                        help='Optimization L2 weight decay [default: 0.0005]')
    parser.add_argument('--t_learning_rate', type=float, default=0.004,
                        help='Initial learning rate for all except decoder [default: 0.004]')
    parser.add_argument('--t_decoder_learning_rate', type=float, default=0.0004,
                        help='Initial learning rate for decoder [default: 0.0004]')
    parser.add_argument('--t_lr_scheduler', type=str, default='step',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--t_lr_decay_epochs', type=int, default=[280, 340], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--t_lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--t_warmup_epoch', type=int, default=-1, help='warmup epoch')
    parser.add_argument('--t_warmup_multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--t_clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)
