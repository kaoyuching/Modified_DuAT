import torch
import torch.nn as nn
import torch.nn.functional as F

from modified_duat.utils import ConvUnit


#############################################
# Selective Boundary Aggregation
#############################################
class RAU(nn.Module):
    r'''
    Re-calibration attention unit (RAU)
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.t1_conv = ConvUnit(in_channels, out_channels, kernel_size=1, padding=0)
        self.t2_conv = ConvUnit(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, t1, t2):
        # input t1
        t1_out = self.t1_conv(t1)
        t1_out = F.sigmoid(t1_out)
        # input t2
        t2_out = self.t2_conv(t2)
        t2_out = F.sigmoid(t2_out)

        # t1_out * t1
        t1_prod = torch.mul(t1_out, t1)
        # t2_out * t2 * (1-t1_out)
        _t2_prod = torch.mul(t2_out, t2)
        # Upsample smaller one
        if t1.shape > t2.shape:
            _t2_prod = F.interpolate(_t2_prod, size=t1.shape[2:], mode='bilinear', align_corners=False)
        else:
            t1_out = F.interpolate(t1_out, size=t2.shape[2:], mode='bilinear', align_corners=False)
            t1_prod = F.interpolate(t1_prod, size=t2.shape[2:], mode='bilinear', align_corners=False)
            t1 = F.interpolate(t1, size=t2.shape[2:], mode='bilinear', align_corners=False)
        t2_prod = torch.mul(_t2_prod, (1 - t1_out))

        out = t1_prod + t2_prod + t1
        return out


class SBABlock(nn.Module):
    r'''
    Selective Boundary Aggregation (SBA)
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.rau_bs = RAU(in_channels, out_channels)
        self.rau_sb = RAU(in_channels, out_channels)
        self.conv = ConvUnit(out_channels*2, out_channels, kernel_size=3, padding=1)

    def forward(self, feature_b, feature_s):
        r'''
        feature_b: features with boundary details. shape: (b, 32, H/4, W/4)
        feature_s: features with semantic information. shape: (b, 32, H/8, W/8)
        '''
        rau_bs = self.rau_bs(feature_b, feature_s)
        rau_sb = self.rau_sb(feature_s, feature_b)
        # concatenation
        cat_rau = torch.cat((rau_bs, rau_sb), dim=1)
        out = self.conv(cat_rau)  # (b, 32, H/4, W/4)
        return out
