# Combining Transformer based Object Detection with the the ScanRefer Pipeline

The overall aim is to improve the results for the visual grounding task.

## TODO

### Open Tasks

important:
- Change coefficients in final loss in loss_helper_detector.py to improve training.
- Add additional layer to fuse module to increase performance (256+228 -> 128)
- referende and detection does not work together. 
- detection does not work at all.
- loss spikes after each epoch (weight decay?)
- sort params in train between refnet and refnetV2
- maybe get rid of refnet?
- make code run with many trans configurations -> dont hard code num layers, etc. in freeze for eg.
- self.detection does not work in solver anymore (bc. not t_freeze...)
- why val box_loss so high?

- why not pick a higher batchsize like 32 instead of 14


additional:
- Change end_points to data_dict globally.
- Implement use_lang_classifier in loss_helper_detector.py.
- Add additional arguments like in Group-Free-3D to detector in refnetV2.py 38.
- Modify ClsAgnosticsPredictHead accordingly to PredictHead.

done:
- Change dimensions of features/querry from [B, 288, 256] to [B, 128, 256] is possible via a MLP.
- Make train.py work.
- Check if objectness mask gets determinated correctly. 
  -> Should be wrong, transformer head predicts objectness score (B, 256, 1) instead of (B, 256, 2).
- Check shapes of added elements from loss_helper_detector and detector.
- Add lr to tensorboard.
- Think about using a pre-trained detector. Maybe for first step pre-trained.
  VoteNet vs. pre-trained detector. Reimplement load_state_dict in eval.py.
- Think if it makes difference that transformer backbone was trained with xyz only.
- Check equivalence to vote_loss in loss_helper_detector.
- Add pretrained transformer weights in eval.py and train.py.
- Make ref_loss drop by creating valid objectness mask in match module.
- Add argparse arguments for detector.
- Implement eval_det in eval.py.




### Discussion/Notes

- ref_acc so low, but IoU so high?
- eval det DONE


## References

### ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language

https://github.com/daveredrum/ScanRefer

### Group-Free 3D Object Detection via Transformers

https://github.com/zeliu98/Group-Free-3D

## Usage / Contribute

To enable the transformer-based object detection add the flag '--transformer'.

Open issues are listed as TODOs in the file 'refnetV2.py'.



