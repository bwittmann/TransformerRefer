import os
import sys
import json
import argparse
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy
from collections import OrderedDict

sys.path.append(os.path.join(os.getcwd()))  # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.config import CONF
from models.refnetV2 import RefNetV2
from models.detector import GroupFreeDetector
from utils.lr_scheduler import get_scheduler

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()

# TODO: for debugging warnings
#np.seterr(all='raise')


def get_dataloader(args, scanrefer, all_scene_list, split):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer[split],
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        augment=args.augment
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    return dataset, dataloader


def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(
        not args.no_height)

    detector_args = {
        'width' : args.width,
        'bn_momentum' : args.bn_momentum,
        'sync_bn' : args.sync_bn,
        'dropout' : args.dropout,
        'activation' : args.activation,
        'nhead' : args.nhead,
        'num_decoder_layers' : args.num_decoder_layers,
        'dim_feedforward' : args.dim_feedforward,
        'cross_position_embedding' : args.cross_position_embedding,
        'size_cls_agnostic' : args.size_cls_agnostic,
        'num_proposals' : args.num_proposals,
        'sampling' : args.sampling,
        'self_position_embedding' : args.self_position_embedding
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
        emb_size=args.emb_size
    )

    # pretrained transformer directly from GroupFreeDetector weights
    if args.use_pretrained_transformer:
        # load model
        print("loading pretrained GroupFreeDetector...")

        pretrained_detector = GroupFreeDetector(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            input_feature_dim=input_channels,
            width= detector_args['width'],
            bn_momentum= detector_args['bn_momentum'], 
            sync_bn= detector_args['sync_bn'], 
            num_proposal=detector_args['num_proposals'],
            sampling=detector_args['sampling'],
            dropout=detector_args['dropout'],
            activation=detector_args['activation'], 
            nhead=detector_args['nhead'], 
            num_decoder_layers=detector_args['num_decoder_layers'],
            dim_feedforward=detector_args['dim_feedforward'], 
            self_position_embedding=detector_args['self_position_embedding'],
            cross_position_embedding=detector_args['cross_position_embedding'],
            size_cls_agnostic=detector_args['size_cls_agnostic']
        )

        # model created with nn.DataParallel -> need to create new ordered dict and remove "module" prefix
        checkpoint = torch.load(args.use_pretrained_transformer, map_location='cpu')
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
        print("loading pretrained ScanRefer with Group Free Transformer detection...")


        pretrained_model = RefNetV2(
            num_class=DC.num_class,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            input_feature_dim=input_channels,
            use_lang_classifier=(not args.no_lang_cls),
            use_bidir=args.use_bidir,
            no_reference=args.no_reference,
            detector_args=detector_args,
            emb_size=args.emb_size
        )

        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        # mount
        model.detector = pretrained_model.detector

    # freeze parts of the detector
    if args.no_detection:
        # freeze complete detector
        for param in model.detector.parameters():
            param.requires_grad = False

    if args.freeze_transformer_layers == 'up_to_pred_heads':
        # freeze complete detector
        for param in model.detector.parameters():
            param.requires_grad = False

        # unfreeze up to prediction heads
        for param in model.detector.prediction_heads.parameters():
            param.requires_grad = True
    elif args.freeze_transformer_layers == 'up_to_decoder_fc':
        # freeze complete detector
        for param in model.detector.parameters():
            param.requires_grad = False

        # unfreeze up to transfomer linear layers (linear1 and linear2)
        for param in model.detector.prediction_heads.parameters():
            param.requires_grad = True

        for layer in range(6):
            for param in model.detector.decoder[layer].linear1.parameters():
                param.requires_grad = True
            for param in model.detector.decoder[layer].linear2.parameters():
                param.requires_grad = True
    elif args.freeze_transformer_layers == 'only_backbone':
        for param in model.detector.backbone_net.parameters():
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
            "lr": args.lr_detector,
            "weight_decay": args.wd_detector},
        {"params": detector_decoder_params,
            "lr": args.lr_detector_decoder,
            "weight_decay": args.wd_detector}
    ]
    optimizer = optim.AdamW(param_dicts,
                            lr=args.lr,
                            weight_decay=args.wd)

    # get learning rate and batch norm scheduler
    lr_scheduler = get_scheduler(optimizer, args.epoch, args.lr_scheduler,
                                    args.lr_decay_epochs, args.lr_decay_rate, args.warmup_epoch,
                                    args.warmup_multiplier, args.lr_patience, args.lr_threshold)
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

    loss_args = {
        "query_points_generator_loss_coef" : args.query_points_generator_loss_coef,
        "obj_loss_coef" : args.obj_loss_coef,
        "box_loss_coef" : args.box_loss_coef,
        "sem_cls_loss_coef" : args.sem_cls_loss_coef,
        "center_delta" : args.center_delta,
        "size_delta" : args.size_delta,
        "heading_delta" : args.heading_delta,
        "detection_loss_coef" : args.detection_loss_coef,
        "ref_loss_coef" : args.ref_loss_coef,
        "lang_loss_coef" : args.lang_loss_coef,
        "use_votenet_objectness" : args.use_votenet_objectness
    }

    # get solver
    solver = Solver(
        model=model,
        config=DC,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        bn_scheduler=bn_scheduler,
        clip_norm=args.clip_norm,
        stamp=stamp,
        no_validation=args.no_validation,
        val_step=args.val_step,
        detection=not args.no_detection,
        reference=not args.no_reference,
        use_lang_classifier=not args.no_lang_cls,
        num_decoder_layers=args.num_decoder_layers,
        loss_args=loss_args
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
    train_dataset, train_dataloader = get_dataloader(args, scanrefer, all_scene_list, "train")
    val_dataset, val_dataloader = get_dataloader(args, scanrefer, all_scene_list, "val")
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
    # general arguments
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--num_workers", type=int, help="number of workers for dataloader", default=4)
    parser.add_argument("--batch_size", type=int, help="batch size", default=14)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)
    parser.add_argument("--num_points", type=int, default=50000, help="point number")
    parser.add_argument("--num_scenes", type=int, default=-1, help="number of scenes")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--augment", action="store_true", help="use data augmentation") 
    parser.add_argument("--no_lang_cls", action="store_true", help="do NOT use language classifier")
    parser.add_argument("--use_bidir", action="store_true", help="use bi-directional GRU")
    parser.add_argument("--emb_size", type=int, default=300, help="input size to GRU")
    parser.add_argument("--no_validation", action="store_true", help="do NOT validate; only for development debugging")


    # input feature arguments
    parser.add_argument("--no_height", action="store_true", help="do NOT use height signal in input")
    parser.add_argument("--use_color", action="store_true", help="use RGB color in input")
    parser.add_argument("--use_normal", action="store_true", help="use normals in input")
    parser.add_argument("--use_multiview", action="store_true", help="use multiview images")


    # optim arguments
    parser.add_argument("--no_reference", action="store_true", help="do NOT train the localization module")
    parser.add_argument("--no_detection", action="store_true", help="do NOT train the detection module")

    parser.add_argument("--use_pretrained", type=str, help="specify the folder name in outputs containing the pretrained model")
    parser.add_argument("--use_pretrained_transformer", type=str, help="specify the absolute file path for pretrained GroupFreeDetector module")
    parser.add_argument("--use_checkpoint", type=str, help="specify the checkpoint root", default="")

    parser.add_argument("--freeze_transformer_layers", type=str, default="none",  help="do NOT train parts of the trans. detection module",
                        choices=["none", "all", "up_to_pred_heads", "up_to_decoder_fc", "only_backbone"])     

    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument('--wd_detector', type=float, default=0.0005, help="L2 weight decay of the detector")

    parser.add_argument("--lr", type=float, help="learning rate of localization part", default=1e-3)
    parser.add_argument('--lr_detector', type=float, default=0.004, help='initial detector learning rate for all except decoder')
    parser.add_argument('--lr_detector_decoder', type=float, default=0.0004, help='initial learning rate for decoder')

    parser.add_argument('--lr_scheduler', type=str, default='step', choices=["step", "cosine", "plateau"], help="learning rate scheduler")

    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340], nargs='+',
                        help='for step scheduler; where to decay lr can be a list (add multiple space-separated values).')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='for step scheduler; decay rate for learning rate')
    parser.add_argument('--lr_patience', type=int, default=10, help='patience for plateau lr scheduler')
    parser.add_argument('--lr_threshold', type=int, default=1e-4, help='measures new optimum for plateau lr scheduler')
    parser.add_argument('--warmup_epoch', type=int, default=0, help='warmup epoch')
    parser.add_argument('--warmup_multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--clip_norm', default=0.1, type=float, help='gradient clipping max norm')


    # loss related arguments
    parser.add_argument('--use_votenet_objectness', action="store_true", help='use objectness as it is used by VoteNet')

    parser.add_argument('--detection_loss_coef', default=1., type=float, help='loss weight for detection loss')  #TODO rename
    parser.add_argument('--ref_loss_coef', default=0.1, type=float, help='loss weight for ref loss')
    parser.add_argument('--lang_loss_coef', default=0.1, type=float, help='loss weight for lang loss')
    parser.add_argument('--query_points_generator_loss_coef', default=0.8, type=float)
    parser.add_argument('--obj_loss_coef', default=0.1, type=float, help='loss weight for objectness loss')
    parser.add_argument('--box_loss_coef', default=1, type=float, help='loss weight for box loss')
    parser.add_argument('--sem_cls_loss_coef', default=0.1, type=float, help='loss weight for classification loss')

    parser.add_argument('--center_delta', default=1.0, type=float, help='delta for smoothl1 loss in center loss')
    parser.add_argument('--size_delta', default=1.0, type=float, help='delta for smoothl1 loss in size loss')
    parser.add_argument('--heading_delta', default=1.0, type=float, help='delta for smoothl1 loss in heading loss')


    # detector related arguments
    parser.add_argument("--num_proposals", type=int, default=256, help="proposal number")
    parser.add_argument("--width", type=int, default=1, help="PointNet backbone width ratio")
    parser.add_argument("--bn_momentum", type=float, default=0.1, help="batchnorm momentum")
    parser.add_argument("--sync_bn", action="store_true", help="converts all bn layers in SyncBatchNorm layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout probability")
    parser.add_argument("--activation", type=str, default='relu', choices=["relu", "gelu", "glu"], help="activation fct used in the decoder layers")
    parser.add_argument("--nhead", type=int, default=8, help="parallel attention heads in multihead attention")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="hidden size of the linear layers in the decoder")
    parser.add_argument("--cross_position_embedding", type=str, default='xyz_learned', choices=["none", "xyz_learned"], 
                        help="position embedding for cross-attention")
    parser.add_argument("--self_position_embedding", type=str, default='loc_learned', choices=["none", "xyz_learned", "loc_learned"], 
                        help="position embedding for self-attention")
    parser.add_argument("--size_cls_agnostic", action="store_true", help="use class agnostic predict heads")
    parser.add_argument("--sampling", type=str, default="kps", help="initial object candidate sampling")


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
