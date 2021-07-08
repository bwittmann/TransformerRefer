import torch
import torch.nn as nn


class MatchModule(nn.Module):
    def __init__(self, num_proposals=256, lang_size=256, hidden_size=128, use_multi_ref_gt=False, num_features=288):
        super().__init__() 

        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size

        self.use_multi_ref_gt = use_multi_ref_gt
        
        self.fuse = nn.Sequential(
            nn.Conv1d(self.lang_size + num_features, hidden_size, 1),
            nn.ReLU()
        )

        self.match = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, 1, 1)
        )

    def forward(self, data_dict):
        # unpack outputs from detection branch
        features = data_dict['last_features'].permute(0, 2, 1).contiguous()

        # predict head outputs objectness_score of shape: batch_size, num_proposals, 1
        objectness_masks = (data_dict['objectness_scores'] > 0).float()  # batch_size, num_proposals, 1

        # unpack outputs from language branch
        lang_feat = data_dict["lang_emb"]  # batch_size, lang_size
        lang_feat = lang_feat.unsqueeze(1).repeat(1, self.num_proposals, 1)  # batch_size, num_proposals, lang_size

        # fuse
        ref_features = torch.cat([features, lang_feat], dim=-1)  # batch_size, num_proposals, 128 + lang_size
        ref_features = ref_features.permute(0, 2, 1).contiguous()  # batch_size, 128 + lang_size, num_proposals

        # fuse features
        ref_features = self.fuse(ref_features)  # batch_size, hidden_size, num_proposals

        if not self.use_multi_ref_gt:
            # mask out invalid proposals
            objectness_masks = objectness_masks.permute(0, 2, 1).contiguous()  # batch_size, 1, num_proposals
            ref_features = ref_features * objectness_masks

        # match
        confidences = self.match(ref_features).squeeze(1)  # batch_size, num_proposals

        if not self.use_multi_ref_gt:
            # NOTE make sure again the empty boxes won't be part of the selection
            confidences.masked_fill_(objectness_masks.squeeze(1) == 0, float('-1e30'))

        data_dict["cluster_ref"] = confidences
        return data_dict
