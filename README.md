# Combining Transformer based Object Detection with the the ScanRefer Pipeline

The overall aim is to improve the results for the visual grounding task.

## TODO

### Open Tasks

important:
- Think about using a pre-trained detector. Maybe for first step pre-trained.
  VoteNet vs. pre-trained detector. Reimplement load_state_dict in eval.py.
- Think if it makes difference that transformer backbone was trained with xyz only.
- Check equivalence to vote_loss in loss_helper_detector.
- Objectness_mask in loss_helper_detector only ones? 
- Change coefficients in final loss in loss_helper_detector.py to improve training.
- Add pretrained transformer weights in eval.py and train.py.
- Make ref_loss drop by creating valid objectness mask in match module.

additional:
- Add argparse arguments for detector.
- Change end_points to data_dict globally.
- Implement use_lang_classifier in loss_helper_detector.py.
- Implement eval_det in eval.py.
- Add additional arguments like in Group-Free-3D to detector in refnetV2.py 38.
- Modify ClsAgnosticsPredictHead accordingly to PredictHead.

done:
- Change dimensions of features/querry from [B, 288, 256] to [B, 128, 256] is possible via a MLP.
- Make train.py work.
- Check if objectness mask gets determinated correctly. 
  -> Should be wrong, transformer head predicts objectness score (B, 256, 1) instead of (B, 256, 2).
- Check shapes of added elements from loss_helper_detector and detector.



### Discussion/Notes

- How to replace vote loss
- pos_ratio, neg_ratio not working
- How to overfitt in an efficient way?
- Use ssh in multiple terminal sessions? (eg background commands)


- Ref_loss does not drop.
  - data_dict["cluster_ref"] are all same value all the time -> same loss
    Error in line 47, objectness_mask only zeros -> last features only zeros


## References

### ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language

https://github.com/daveredrum/ScanRefer

### Group-Free 3D Object Detection via Transformers

https://github.com/zeliu98/Group-Free-3D

## Usage / Contribute

To enable the transformer-based object detection add the flag '--transformer'.

Open issues are listed as TODOs in the file 'refnetV2.py'.



