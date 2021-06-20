import torch.nn as nn
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.lang_module import LangModule
from models.match_module import MatchModule
from models.detector import GroupFreeDetector


class RefNetV2(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
                 input_feature_dim, use_lang_classifier, use_bidir, emb_size,
                 detector_args, no_reference=False, hidden_size=256, use_multi_ref_gt=False):
        super().__init__()

        assert(mean_size_arr.shape[0] == num_size_cluster)    
        self.no_reference = no_reference


        # ---------- TRANSFORMER ------------
        self.detector = GroupFreeDetector(
            num_class=num_class,
            num_heading_bin=num_heading_bin,
            num_size_cluster=num_size_cluster,
            mean_size_arr=mean_size_arr,
            input_feature_dim=input_feature_dim,
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
            size_cls_agnostic=detector_args['size_cls_agnostic'],
            num_features=detector_args['num_features']
        )
        
        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, hidden_size)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(
                num_proposals=detector_args['num_proposals'], lang_size=(1 + int(use_bidir)) * hidden_size, 
                use_multi_ref_gt=use_multi_ref_gt, num_features=detector_args['num_features']
            )

    def forward(self, data_dict):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- TRANSFORMER ---------
        data_dict = self.detector(data_dict)

        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang(data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            data_dict = self.match(data_dict)

        return data_dict
