import torch
import torch.nn as nn
import torchvision.models as models


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        self.visual_extractor = args.visual_extractor
        self.pretrained = args.visual_extractor_pretrained
        model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
        modules = list(model.children())[:-2]  # 去除 avgpool 和 fc
        self.model = nn.Sequential(*modules)
        self.avg_fnt = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images):
        feature_map = self.model(images)                 # [B, C, H, W]
        fc_feats = self.avg_fnt(feature_map).flatten(1)  # [B, C]

        patch_feats = feature_map
        return patch_feats, fc_feats