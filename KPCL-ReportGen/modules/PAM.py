import torch
import torch.nn as nn
import torch.nn.functional as F



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None: x = self.bn(x)
        if self.relu is not None: x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], keyword_dim=None):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.pool_types = pool_types
        self.use_keyword = keyword_dim is not None
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        if self.use_keyword:
            self.keyword_proj = nn.Linear(keyword_dim, gate_channels)

    def forward(self, x, keyword=None):
        batch_size, C, H, W = x.size()
        channel_att_sum = 0
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, C)
            elif pool_type == 'max':
                pool = F.adaptive_max_pool2d(x, 1).view(batch_size, C)
            else:
                raise NotImplementedError
            channel_att_sum += self.mlp(pool)

        if self.use_keyword and keyword is not None:
            keyword_bias = self.keyword_proj(keyword)  # [B, C]
            channel_att_sum += keyword_bias

        scale = torch.sigmoid(channel_att_sum).view(batch_size, C, 1, 1)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flat = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flat, dim=2, keepdim=True)
    outputs = s + (tensor_flat - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
class SpatialGate(nn.Module):
    def __init__(self, keyword_dim=None):
        super(SpatialGate, self).__init__()
        self.compress = lambda x: torch.cat(
            (torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)), dim=1)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.use_keyword = keyword_dim is not None
        if self.use_keyword:
            self.keyword_proj = nn.Linear(keyword_dim, 1)

    def forward(self, x, keyword=None):
        x_compress = self.compress(x)  # [B, 2, H, W]
        attention = self.spatial(x_compress)  # [B, 1, H, W]

        if self.use_keyword and keyword is not None:
            B, _, H, W = attention.size()
            keyword_bias = self.keyword_proj(keyword).unsqueeze(2).unsqueeze(3).expand(B, 1, H, W)
            attention += keyword_bias

        scale = torch.sigmoid(attention)
        return x * scale


class ParallelAttention_KG(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], keyword_dim=None):
        super(ParallelAttention_KG, self).__init__()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio, pool_types, keyword_dim)
        self.spatial_gate = SpatialGate(keyword_dim)
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x, keyword=None):
        x_channel = self.channel_gate(x, keyword)
        x_spatial = self.spatial_gate(x, keyword)
        out = self.alpha * x_channel + self.beta * x_spatial
        return out
