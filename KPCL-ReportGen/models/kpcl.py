import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.PAM import ParallelAttention_KG
from modules.mgcl import MGCL


class KPCLModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(KPCLModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

        self.att_module = ParallelAttention_KG(
            gate_channels=args.d_vf,
            reduction_ratio=16,
            pool_types=['avg', 'max'],
            keyword_dim = args.keyword_dim
        )
        # 加载关键词嵌入
        if args.mode == 'train':
            self.keyword_embeddings = torch.load('C:/Users/Admin/Desktop/KPCL-ReportGen/data/PED_xray/keyword_vecs/train_keywords.pt')  # [N, 512]

        # MGCL 模块（局部-全局图损失）
        self.mgcl = MGCL(dim_hidden=args.d_vf * 2)  # 输入为拼接后的 2C 维度

        if args.dataset_name == 'PED_xray':
            self.forward = self.forward_PED_xray
        else:
            raise ValueError(f"Unknown dataset_name: {args.dataset_name}. Expected 'PED_xray'.")



    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_PED_xray(self, images, targets=None, image_ids=None, mode='train'):
        # 图像特征提取（正位 & 侧位）
        att_feats_0_raw, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1_raw, fc_feats_1 = self.visual_extractor(images[:, 1])

        batch_size = images.size(0)

        # 拼接成全局和局部特征（用于 MGCL）
        global_feat = torch.cat((fc_feats_0, fc_feats_1), dim=1)          # (B, 2C)
        patch_feat = torch.cat((att_feats_0_raw, att_feats_1_raw), dim=1) # (B, 2C, H, W)

        if mode == 'train':
            graph_loss = self.mgcl(global_feat, patch_feat, image_ids=image_ids)
        else:
            graph_loss = None

        # 并行注意力引导
        att_feats_0 = self.att_module(att_feats_0_raw)
        att_feats_1 = self.att_module(att_feats_1_raw)

        # 展平为 patch token（B, C, H, W）→ (B, N, C)
        att_feats_0 = att_feats_0.flatten(2).permute(0, 2, 1)
        att_feats_1 = att_feats_1.flatten(2).permute(0, 2, 1)

        # 拼接正位 & 侧位
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        # 解码生成报告
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError

        return output, graph_loss

