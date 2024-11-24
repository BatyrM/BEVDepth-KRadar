import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SECONDFPN(nn.Module):
    """
        This module implements SECOND FPN, which creates pyramid features built on top of some input feature maps.
        This code is adapted from https://github.com/open-mmlab/mmdetection3d/blob/main/mmdet3d/models/necks/second_fpn.py with some modifications".
    """
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        in_channels =  self.model_cfg.IN_CHANNELS
        out_channels = self.model_cfg.OUT_CHANNELS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(in_channels) == len(out_channels) == len(upsample_strides)
        use_bias = self.model_cfg.get('USE_BIAS', False)
        use_conv_for_no_stride = self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)
        self.in_channels = in_channels
        self.out_channels = out_channels

        deblocks = []

        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = nn.ConvTranspose2d(in_channels=in_channels[i], out_channels=out_channel, kernel_size=upsample_strides[i], stride=upsample_strides[i], bias=use_bias)
            else:
                stride = np.round(1/stride).astype(np.int64)
                upsample_layer = nn.Conv2d(in_channels=in_channels[i], out_channels=out_channel, kernel_size=stride, stride=stride, bias=use_bias)
            
            deblock = nn.Sequential(upsample_layer, nn.BatchNorm2d(num_features=out_channel, eps=1e-3, momentum=0.01), nn.ReLU(inplace=True))

            deblocks.append(deblock)
        
        self.deblocks = nn.ModuleList(deblocks)

    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(tensor=m.weight, a=0, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, batch_dict):
        x = batch_dict['image_features']
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]

        batch_dict['image_fpn'] = [out]

        return batch_dict

