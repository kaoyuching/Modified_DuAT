import torch
import torch.nn as nn
import torch.nn.functional as F


#############################################
# GLAM module
# Global-local, spatial-channel attention
#############################################
# Global attention
class GlobalChannelAttention(nn.Module):
    r'''
    Global channel attention(GCA)
    '''
    def __init__(self, feature_map_size, kernel_size=1):
        super().__init__()
        self.gap = nn.AvgPool2d(feature_map_size)
        self.conv_q = nn.Conv1d(1, 1, kernel_size, stride=1, padding=(kernel_size - 1)//2)
        self.conv_k = nn.Conv1d(1, 1, kernel_size, stride=1, padding=(kernel_size - 1)//2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature):
        r'''
        feature: shape (B, C, H, W)
        1. Apply global average pooling(GAP) and squeeze spatial dimensions.
        2. Apply 1D convolution of kernel size k and a sigmoid function.
        3. Get 1*c query(Q) and key(K).
        4. Get value(V) by reshaping feature to hw*c without GAP.
        5. Perform the outer product of K and Q, followed by softmax over channels,
        and obtain c*c global channel attention map.
        Att = softmax(K^TQ)
        6. Obtain attention feature map by performing matrix product of V and Att and
        reshape back to (B, C, H, W)
        '''
        b, c, h, w = feature.shape
        feature_gap = self.gap(feature).reshape(b, 1, c)  # (b, c, 1, 1) -> (b, 1, c)
        query = F.sigmoid(self.conv_q(feature_gap))  # (b, 1, c)
        key = F.sigmoid(self.conv_k(feature_gap))  # (b, 1, c)
        value = feature.reshape(b, h*w, c)  # (b, h*w, c)
        att = torch.matmul(key.transpose(1, 2), query)  # (b, c, c)
        att = self.softmax(att)
        feature_gc = torch.matmul(value, att)  # (b, h*w, c)
        feature_gc = feature_gc.reshape(b, c, h, w)
        return feature_gc


class GlobalSpatialAttention(nn.Module):
    r'''
    global spacial attention(GSA)
    '''
    def __init__(self, in_channels, embedding_num):
        super().__init__()
        self.emb_num = embedding_num
        self.conv_q = nn.Conv2d(in_channels, embedding_num, 1)
        self.conv_k = nn.Conv2d(in_channels, embedding_num, 1)
        self.conv_v = nn.Conv2d(in_channels, embedding_num, 1)
        self.softmax = nn.Softmax(dim=2)  # softmax over location
        self.conv_att = nn.Conv2d(embedding_num, in_channels, 1)

    def forward(self, feature):
        r'''
        feature: shape (B, C, H, W)
        1. Obtain Q, K, V by applying 1*1 convolution reducing channel to c', and flatening
        spatial dimensions to hw. Q, K, V with shape (B, c', HW)
        2. Obtain global spatial attention: Att = softmax(K^TQ) with shape(B, HW, HW)
        3. Obtain global spatial attention map by performing matrix product of V and Att and
        reshape back to (B, c', H, W).
        4. Increase channel back to C with 1*1 convolution.
        '''
        b, c, h, w = feature.shape
        query = self.conv_q(feature).reshape(b, self.emb_num, h*w)  # (b, c', hw)
        key = self.conv_k(feature).reshape(b, self.emb_num, h*w)  # (b, c', hw)
        value = self.conv_v(feature).reshape(b, self.emb_num, h*w)  # (b, c', hw)
        att = torch.matmul(key.transpose(1, 2), query)  # (b, hw, hw)
        att = self.softmax(att)
        feature_gs = torch.matmul(value, att)  # (b, c', hw)
        feature_gs = feature_gs.reshape(b, self.emb_num, h, w)
        feature_gs = self.conv_att(feature_gs)  # (b, c, h, w)
        return feature_gs


class GlobalAttention(nn.Module):
    r'''
    Global attention module.
    '''
    def __init__(self, feature_map_size, in_channels, embedding_num, kernel_size=1):
        super().__init__()
        self.gca = GlobalChannelAttention(feature_map_size, kernel_size=kernel_size)
        self.gsa = GlobalSpatialAttention(in_channels, embedding_num)

    def forward(self, feature):
        gc = self.gca(feature)  # (b, c, h, w)
        gs = self.gsa(feature)  # (b, c, h, w)
        feature_gc = feature * gc
        feature_global = feature_gc * gs + feature_gc
        return feature_global


# local attention
class LocalChannelAttention(nn.Module):
    r'''
    Local channel attention.
    '''
    def __init__(self, feature_map_size, in_channels, kernel_size=1):
        super().__init__()
        self.gap = nn.AvgPool2d(feature_map_size)
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=(kernel_size - 1)//2)

    def forward(self, feature):
        r'''
        feature: shape (B, C, H, W)
        1. Apply global average pooling to reduce tensor to shape (B, C, 1, 1)
        '''
        b, c, h, w = feature.shape
        feature_gap = self.gap(feature)  # (b, c, 1, 1)
        feature_lc = self.conv(feature_gap.reshape(b, c, 1)).reshape(b, c, 1, 1)
        feature_lc = F.sigmoid(feature_lc)
        return feature_lc


class LocalSpatialAttention(nn.Module):
    r'''
    Local spatial attention.
    '''
    def __init__(self, in_channels, embedding_num):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, embedding_num, 1)
        self.conv3 = nn.Conv2d(embedding_num, embedding_num, 3, padding=1, dilation=1)
        self.conv5 = nn.Conv2d(embedding_num, embedding_num, 3, padding=2, dilation=2)
        self.conv7 = nn.Conv2d(embedding_num, embedding_num, 3, padding=3, dilation=3)
        self.conv_ls = nn.Conv2d(4*embedding_num, 1, 1)

    def forward(self, feature):
        b, c, h, w = feature.shape
        f1 = self.conv1(feature)  # (b, c', h, w)
        f3 = self.conv3(f1)  # (b, c', h, w)
        f5 = self.conv5(f1)  # (b, c', h, w)
        f7 = self.conv7(f1)  # (b, c', h, w)
        fea_concat = torch.cat((f1, f3, f5, f7), 1)  # (b, 4c', h, w)
        feature_ls = self.conv_ls(fea_concat)  # (b, 1, h, w)
        return feature_ls


class LocalAttention(nn.Module):
    r'''
    Local attention module.
    '''
    def __init__(self, feature_map_size, in_channels, embedding_num, kernel_size=1):
        super().__init__()
        self.lca = LocalChannelAttention(feature_map_size, in_channels, kernel_size=kernel_size)
        self.lsa = LocalSpatialAttention(in_channels, embedding_num)

    def forward(self, feature):
        att_c = self.lca(feature)  # (b, c, 1, 1)
        att_s = self.lsa(feature)  # (b, 1, h, w)
        feature_lc = feature * att_c + feature  # (b, c, h, w)
        feature_local = feature_lc * att_s + feature_lc  # (b, c, h, w)
        return feature_local


class GLAM(nn.Module):
    r'''
    Global-Local Attention Module (GLAM).
    '''
    def __init__(self, feature_map_size, in_channels, out_channels, embedding_num, kernel_size=1):
        super().__init__()
        self.local_att_fn = LocalAttention(feature_map_size, in_channels, embedding_num, kernel_size)
        self.global_att_fn = GlobalAttention(feature_map_size, in_channels, embedding_num, kernel_size)
        self.fusion_weight = nn.Parameter(torch.Tensor([1/3, 1/3, 1/3]))
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, feature):
        orig = feature.unsqueeze(1)  # (b, 1, c, h, w)
        local_att = self.local_att_fn(feature).unsqueeze(1)  # (b, 1, c, h, w)
        global_att = self.global_att_fn(feature).unsqueeze(1)  # (b, 1, c, h, w)
        # feature fusion: weighted average
        weight = F.softmax(self.fusion_weight, dim=None).reshape(1, 3, 1, 1, 1)
        fusion_feature = weight * torch.cat((local_att, global_att, orig), dim=1)
        fusion_feature = torch.sum(fusion_feature, dim=1)  # (b, c, h, w)
        fusion_feature = self.conv(fusion_feature)  # (b, c_out, h, w)
        return fusion_feature
