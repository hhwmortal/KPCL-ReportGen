import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import numpy as np
from sentence_transformers import SentenceTransformer
import json

class ProjectionHead(nn.Module):
    def __init__(self, dim_in=2048, dim_out=2048, dim_hidden=2048):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_hidden)
        self.ln1 = nn.LayerNorm(dim_hidden)
        self.relu1 = nn.ReLU(True)
        self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        self.ln2 = nn.LayerNorm(dim_hidden)
        self.relu2 = nn.ReLU(True)
        self.linear3 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.ln1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x



class MGCL(nn.Module):
    def __init__(self, dim_hidden=2048, report_json_path='C:/Users/Admin/Desktop/KPCL-ReportGen/data/PED_xray/annotation.json'):
        super().__init__()
        self.proj_global = ProjectionHead(dim_in=dim_hidden, dim_out=dim_hidden, dim_hidden=dim_hidden)
        self.proj_local = ProjectionHead(dim_in=dim_hidden, dim_out=dim_hidden, dim_hidden=dim_hidden)
        self.sbert = SentenceTransformer('C:/Users/Admin/Desktop/KPCL-ReportGen/models/distiluse-base-multilingual-cased')

        # 读取 JSON 文件
        self.report_dict = {}
        with open(report_json_path, 'r', encoding='utf-8-sig') as f:
            raw_data = json.load(f)
            for entry in raw_data['train'] + raw_data.get('val', []) + raw_data.get('test', []):
                self.report_dict[entry['id']] = entry['report']

    def get_report_texts(self, image_ids):
        processed_ids = [img_id.split('/')[-1].replace("IMG-0001-", "").replace("IMG-0002-", "") for img_id in image_ids]
        return [self.report_dict[pid] for pid in processed_ids]

    @torch.no_grad()
    def build_text_guided_graph(self, img_feat, text_feat, threshold=0.7):
        sim_img = F.cosine_similarity(img_feat.unsqueeze(1), img_feat.unsqueeze(0), dim=-1)  # [B, B]
        sim_txt = F.cosine_similarity(text_feat.unsqueeze(1), text_feat.unsqueeze(0), dim=-1)

        mask = (sim_img > threshold) & (sim_txt > threshold)  # 同时满足
        b = img_feat.size(0)
        graph = csr_matrix(mask.cpu().numpy())
        _, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        labels = torch.tensor(labels, device=img_feat.device)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        return mask  # [B, B]

    def sup_contra(self, logits, mask):
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = (-mean_log_prob_pos).mean()
        return loss

    def forward(self, global_feat, patch_feat, image_ids, t=0.1, alpha=0.5):
        """
        global_feat: (B, C)
        patch_feat:  (B, C, H, W)
        image_ids: list[str]  # e.g., ['image00001/IMG-0001-00001.jpg', ...]
        """
        device = global_feat.device
        report_texts = self.get_report_texts(image_ids)  # 从 JSON 中获取报告

        with torch.no_grad():
            text_embed = self.sbert.encode(report_texts, convert_to_tensor=True).to(device)

        # 2. Projection
        feat_g = F.normalize(self.proj_global(global_feat), dim=-1)  # [B, D]
        feat_l = F.normalize(self.proj_local(patch_feat.mean(dim=[2, 3])), dim=-1)  # [B, D]

        # 3. 生成图结构 mask
        with torch.no_grad():
            mask_global = self.build_text_guided_graph(feat_g, text_embed)  # [B, B]
            mask_local = self.build_text_guided_graph(feat_l, text_embed)  # [B, B]

        # 4. Cosine similarity logits
        logit_g = torch.mm(feat_g, feat_g.T) / t  # [B, B]
        logit_l = torch.mm(feat_l, feat_l.T) / t

        # 5. 计算两个对比损失
        loss_g = self.sup_contra(logit_g, mask_global)
        loss_l = self.sup_contra(logit_l, mask_local)

        # 6. 融合
        graph_loss = alpha * loss_g + (1 - alpha) * loss_l
        return graph_loss







