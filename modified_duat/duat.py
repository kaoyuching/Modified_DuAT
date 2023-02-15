import warnings
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from modified_duat.utils import ConvUnit
from modified_duat.pvt_encoder import PyramidVisionTransformerV2
from modified_duat.glam import GLAM
from modified_duat.sba import SBABlock


r"""
DuAT: PVT encoder, SBA module, GLSA module
"""


class DuAT(nn.Module):
    r'''
    model DuAT

    Args:
        - in_channels(int): number of input channels. Default: 3.
        - encoder_name(str): one of {'pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3',
        'pvt_v2_b4', 'pvt_v2_b5', 'pvt_v2_b2_li'}.
        - encoder_pretrained(optional, str): pretrained weight path of encoder. Default: None.
        - in_feature_size(int): input image size. Default: 512.
        - num_classes(int): output channel's number. Default: 1.
    '''
    def __init__(
            self,
            in_channels: int = 3,
            encoder_name: str = 'pvt_v2_b2',
            encoder_pretrained: Optional[str] = None,
            in_feature_size: int = 512,
            num_classes: int = 1,
        ):
        super().__init__()
        # encoder: PVT
        if encoder_name.lower() not in ['pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5', 'pvt_v2_b2_li']:
            raise ValueError(f"encoder_name should be one of ['pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2', 'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5', 'pvt_v2_b2_li']. Got {encoder_name}.")
        self.encoder = create_model(encoder_name.lower(), in_chans=in_channels)
        if encoder_pretrained is not None:
            try:
               checkpoint = torch.load(encoder_pretrained, map_location='cpu')
               param = checkpoint['patch_embed1.proj.weight']
               if param.shape[1] != in_channels:
                   new_param = torch.mean(param, 1, keepdim=True).repeat((1, in_channels, 1, 1))
                   new_param = nn.parameter.Parameter(new_param, requires_grad=True)
                   checkpoint['patch_embed1.proj.weight'] = new_param
               self.encoder.load_state_dict(checkpoint)
            except:
               warnings.warn("No pretrained used. Use init weights.")
        # encoder's embedding dims
        self.embed_dims = self.encoder.embed_dims
        # SBA params: in_channels, out_channels
        self.sba = SBABlock(32, 32)
        # GLAM params: feature_map_size, in_channels, out_channels, embedding_num, kernel_size
        self.glam2 = GLAM(
            in_feature_size//8,
            self.embed_dims[1],
            32,
            self.embed_dims[1]//2,
            kernel_size=3
        )
        self.glam3 = GLAM(
            in_feature_size//16,
            self.embed_dims[2],
            32,
            self.embed_dims[2]//2,
            kernel_size=3
        )
        self.glam4 = GLAM(
            in_feature_size//32,
            self.embed_dims[3],
            32,
            self.embed_dims[3]//2,
            kernel_size=3
        )

        self.f1_conv = ConvUnit(self.embed_dims[0], 32, kernel_size=3, padding=1)
        self.glam_conv1 = ConvUnit(64, 32, kernel_size=1)  # for SBA module input
        # glam output conv
        self.glam_conv2 = ConvUnit(96, num_classes, kernel_size=1)
        # sba output conv
        self.sba_conv = ConvUnit(32, num_classes, kernel_size=1)

    def forward(self, feature):
        b, c, h, w = feature.shape
        # PVT
        f1, f2, f3, f4 = self.encoder.forward_features(feature)
        # GLAM
        f2_out = self.glam2(f2)  # (b, 32, H/8, W/8)
        f3_out = self.glam3(f3)  # (b, 32, H/16, W/16)
        f3_out = F.upsample(f3_out, size=f2_out.shape[2:], mode='bilinear', align_corners=False)  # (b, 32, H/8, W/8)
        f4_out = self.glam4(f4)  # (b, 32, H/32, W/32)
        f4_out = F.upsample(f4_out, size=f2_out.shape[2:], mode='bilinear', align_corners=False)  # (b, 32, H/8, W/8)
        feature_glam = torch.cat((f2_out, f3_out, f4_out), dim=1)  # (b, 96, H/8, W/8)
        feature_glam = self.glam_conv2(feature_glam)  # (b, num_classes, H/8, W/8)
        feature_glam = F.upsample(feature_glam, size=(h, w), mode='bilinear', align_corners=False)
        # SBA
        feature_b = self.f1_conv(f1)  # (b, 32, H/4, W/4)
        feature_s = torch.cat((f3_out, f4_out), dim=1)  # (b, 64, H/8, W/8)
        feature_s = self.glam_conv1(feature_s)  # (b, 32, H/8, W/8)
        feature_sba = self.sba(feature_b, feature_s)  # (b, 32, H/4, W/4)
        feature_sba = self.sba_conv(feature_sba)  # (b, num_classes, H/4, W/4)
        feature_sba = F.upsample(feature_sba, size=(h, w), mode='bilinear', align_corners=False)
        return feature_glam, feature_sba
