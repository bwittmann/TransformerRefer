# RefNetV2 - 3D visual grounding using a transformer-based object detector

Combining Transformer based Object Detection with the the ScanRefer Pipeline.
The overall aim is to improve the results for the visual grounding task.



## Usage / Contribute



## References

### ScanRefer: 3D Object Localization in RGB-D Scans using Natural Language

https://github.com/daveredrum/ScanRefer

### Group-Free 3D Object Detection via Transformers

https://github.com/zeliu98/Group-Free-3D


# TODO:
- rename get_loss_detector to get_loss
- clean visualize.py
- rename args from train.py

additional:
- Change end_points to data_dict globally.
- Implement use_lang_classifier in loss_helper_detector.py.
- Add additional arguments like in Group-Free-3D to detector in refnetV2.py 38.
- Modify ClsAgnosticsPredictHead accordingly to PredictHead.


