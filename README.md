# Combining Transformer based Object Detection with the the ScanRefer Pipeline

The overall aim is to improve the results for the visual grounding task.

## Open Tasks

- Add argparse arguments for detector.
- Change dimensions of features/querry from [B, 288, 256] to [B, 128, 256]
  is possible via a MLP. DONE
- Think about using a pre-trained detector. Maybe for first step pre-trained.
  VoteNet vs. pre-trained detector. Reimplement load_state_dict in eval.py.
- Make train.py work.
- Think if it makes difference that transformer backbone was trained with xyz only.
- Check if objectness mask gets determinated correctly. 
  -> Should be correct, transformer head predicts objectness score (B, 256, 1) instead of (B, 256, 2)
- Check shapes of added elements from loss_helper_detector and detector. DONE 
- Check equivalence to vote_loss in loss_helper_detector.
- Objectness_mask in loss only ones? 
- Change end_points to data_dict globally.
- Implement use_lang_classifier in loss_helper_detector.py.
- Change coefficients in final loss in loss_helper_detector.py.
- Add pretrained transformer weights in eval.py.

## References

### ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language

https://github.com/daveredrum/ScanRefer

### Group-Free 3D Object Detection via Transformers

https://github.com/zeliu98/Group-Free-3D

## Usage / Contribute

To enable the transformer-based object detection add the flag '--transformer'.

Open issues are listed as TODOs in the file 'refnetV2.py'.



