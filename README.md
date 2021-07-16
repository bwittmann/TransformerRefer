# RefNetV2 - 3D visual grounding using a transformer-based object detector

In this project the ScanRefer baseline architecture is modified to make use of improved object detector.
The overall aim is to improve the results for the visual grounding task.


## Usage

### training

Object detector with 6 transformer decoder blocks:

    python scripts/train.py --no_lang_cls --batch_size 10 --epoch 200 --lr_scheduler plateau --augment --val_step 3000

Object detector with 12 transformer decoder blocks:

    python scripts/train.py --epoch 200 --lr_scheduler plateau --augment --val_step 3000 --num_decoder_layers 12

Object detector with 12 transformer decoder blocks and a double width backbone:

    python scripts/train.py  --epoch 200 --lr_scheduler plateau --augment --val_step 3000 --num_decoder_layers 12 --width 2

It is strongly adviced to use a pre-trained weight initialization for the transformer-based object detector:

    --use_pretrained_transformer <path to model>

Additional input features, select appropriate flags:

    --use_colors --use_normal --use_multiview --no_height

Language classification proxy loss:

    --use_lang_cls

Multi-ref loss:

    --use_multi_ref_gt

### eval

    python scripts/eval.py --folder <path to model> --reference --no_nms --force --repeat 5

### visualize
    python scripts/visualize.py --folder <path to model> --scene_id <id from val set> 

### further information
    python scripts/train.py --help
    python scripts/eval.py --help
    python scripts/visualize.py --help

## Pre-trained models

Please contact bastian.wittmann@tum.de or philipp.foth@tum.de in order to receive pre-trained models

## Dataset

To setup and preparate the data, please refer to the <a href="https://github.com/daveredrum/ScanRefer" target="_blank">ScanRefer</a> repository.

## Citation

The project and source code is based on the  <a href="https://github.com/daveredrum/ScanRefer" target="_blank">ScanRefer</a>  and the  <a href="https://github.com/zeliu98/Group-Free-3D" target="_blank">Group-Free-3D</a> project.

    @article{chen2020scanrefer,
        title={ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language},
        author={Chen, Dave Zhenyu and Chang, Angel X and Nie{\ss}ner, Matthias},
        journal={16th European Conference on Computer Vision (ECCV)},
        year={2020}
    }

    @inproceedings{dai2017scannet,
        title={Scannet: Richly-annotated 3d reconstructions of indoor scenes},
        author={Dai, Angela and Chang, Angel X and Savva, Manolis and Halber, Maciej and Funkhouser, Thomas and Nie{\ss}ner, Matthias},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
        pages={5828--5839},
        year={2017}
    }

    @article{liu2021,
        title={Group-Free 3D Object Detection via Transformers},
        author={Liu, Ze and Zhang, Zheng and Cao, Yue and Hu, Han and Tong, Xin},
        journal={arXiv preprint arXiv:2104.00678},
        year={2021}
    }

