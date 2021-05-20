import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
from models.proposal_module import ProposalModule
from models.lang_module import LangModule
from models.match_module import MatchModule
from models.detector import GroupFreeDetector


# TODO: 
# Add argparse arguments for detector.
# Change dimensions of features/querry from [B, 288, 256] to [B, 128, 256]
#   is possible via a MLP. DONE
# Think about using a pre-trained detector. Maybe for first step pre-trained.
#   VoteNet vs. pre-trained detector. Reimplement load_state_dict in eval.py.
# Make train.py work.
# Think if it makes difference that transformer backbone was trained with xyz only.
# Check if objectness mask gets determinated correctly. 
#   -> Should be correct, transformer head predicts objectness score (B, 256, 1) instead of (B, 256, 2)
# Check shapes of added elements from loss_helper_detector and detector. DONE 
# Check equivalence to vote_loss in loss_helper_detector.
# Objectness_mask in loss only ones? 
# Change end_points to data_dict globally.
# Implement use_lang_classifier in loss_helper_detector.py.
# Change coefficients in final loss in loss_helper_detector.py.
# Add pretrained transformer weights in eval.py.


class RefNetV2(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, 
    input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
    use_lang_classifier=True, use_bidir=False, no_reference=False,
    emb_size=300, hidden_size=256):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir      
        self.no_reference = no_reference


        # ---------- TRANSFORMER ------------
        self.detector = GroupFreeDetector(num_class=num_class,
                              num_heading_bin=num_heading_bin,
                              num_size_cluster=num_size_cluster,
                              mean_size_arr=mean_size_arr,
                              input_feature_dim=input_feature_dim,
                              num_proposal=num_proposal,
                              self_position_embedding='loc_learned')
        
        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, hidden_size)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * hidden_size, use_trans=True)

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
