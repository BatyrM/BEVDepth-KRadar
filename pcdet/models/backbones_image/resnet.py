import logging
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        dilation=dilation,
        bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None):

        super(BasicBlock, self).__init__()

        self.norm1_name, norm1 = 'bn1', nn.BatchNorm2d(num_features=planes)
        self.norm2_name, norm2 = 'bn2', nn.BatchNorm2d(num_features=planes)

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv3x3(planes, planes)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation

        self.norm1_name, norm1 = 'bn1', nn.BatchNorm2d(num_features=planes)
        self.norm2_name, norm2 = 'bn2', nn.BatchNorm2d(num_features=planes)
        self.norm3_name, norm3 = 'bn3', nn.BatchNorm2d(num_features=planes * self.expansion)

        self.conv1 = conv1x1(inplanes, planes)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv3x3(planes, planes, stride=stride, dilation=dilation)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out

class ResLayer(nn.Sequential):
    """ResLayer to build ResNet style backbone.

    Args:
        block (nn.Module): block used to build ResLayer.
        inplanes (int): inplanes of block.
        planes (int): planes of block.
        num_blocks (int): number of blocks.
        stride (int): stride of the first block. Default: 1
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
        downsample_first (bool): Downsample at the first block or last block.
            False for Hourglass, True for ResNet. Default: True
    """

    def __init__(self,
                 block,
                 inplanes,
                 planes,
                 num_blocks,
                 stride=1,
                 avg_down=False,
                 downsample_first=True,
                 dilation=1):

        self.block = block

        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = []
            conv_stride = stride
            if avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool2d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False))
            downsample.extend([
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=conv_stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            ])
            downsample = nn.Sequential(*downsample)

        layers = []
        if downsample_first:
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    dilation=dilation))
            inplanes = planes * block.expansion
            for _ in range(1, num_blocks):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=planes,
                        stride=1,
                        dilation=dilation))

        else:  # downsample_first=False is for HourglassModule
            for _ in range(num_blocks - 1):
                layers.append(
                    block(
                        inplanes=inplanes,
                        planes=inplanes,
                        stride=1,
                        dilation=dilation))
            layers.append(
                block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=stride,
                    downsample=downsample,
                    dilation=dilation))
        super(ResLayer, self).__init__(*layers)

class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 model_cfg,
                 in_channels=3,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=[0, 1, 2, 3],
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 zero_init_residual=True):

        super(ResNet, self).__init__()

        self.model_cfg = model_cfg
        self.in_channels = self.model_cfg.get('IN_CHANNELS', in_channels)
        self.zero_init_residual = self.model_cfg.get('ZERO_INIT_RESIDUAL', zero_init_residual)
        self.depth = self.model_cfg.get('DEPTH', None)
        if self.depth not in self.arch_settings:
            raise KeyError(f'invalid depth {self.depth} for resnet')
        self.base_channels = self.model_cfg.get('BASE_CHANNELS', base_channels)
        self.stem_channels = self.model_cfg.get('STEM_CHANNELS', self.base_channels)
        self.num_stages = self.model_cfg.get('NUM_STAGES', num_stages)
        assert self.num_stages >= 1 and self.num_stages <= 4
        self.strides = self.model_cfg.get('STRIDES', strides)
        self.strides = tuple(self.strides) if isinstance(self.strides, list) else self.strides
        self.dilations = self.model_cfg.get('DILATIONS', dilations)
        self.dilations = tuple(self.dilations) if isinstance(self.dilations, list) else self.dilations
        assert len(self.strides) == len(self.dilations) == self.num_stages
        self.out_indices = self.model_cfg.get('OUT_INDICES', out_indices)
        assert max(self.out_indices) < self.num_stages
        self.deep_stem = self.model_cfg.get('DEEP_STEM', deep_stem)
        self.avg_down = self.model_cfg.get('AVG_DOWN', avg_down)
        self.frozen_stages = self.model_cfg.get('FROZEN_STAGES', frozen_stages)
        self.norm_eval = self.model_cfg.get('NORM_EVAL', norm_eval)
        self.dcn = self.model_cfg.get('DCN', dcn)
        self.stage_with_dcn = self.model_cfg.get('STAGE_WITH_DCN', stage_with_dcn)
        if dcn is not None:
            assert len(self.stage_with_dcn) == self.num_stages
        self.block, stage_blocks = self.arch_settings[self.depth]
        self.stage_blocks = stage_blocks[:self.num_stages]
        self.inplanes = self.stem_channels
        self.pretrained = self.model_cfg.get('PRETRAINED', False)
        self.checkpoint = self.model_cfg.get('CHECKPOINT', None)

        self._make_stem_layer(self.in_channels, self.stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                avg_down=self.avg_down,
                dilation=dilation)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * self.base_channels * 2**(
            len(self.stage_blocks) - 1)

    def init_weights(self):
        if self.pretrained and self.checkpoint is not None:
            self.load_state_dict(torch.load(self.checkpoint), strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                        nn.init.constant_(m.bn2.weight, 0)

    def make_res_layer(self, block, inplanes, planes, num_blocks, stride, avg_down, dilation):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(block, inplanes, planes, num_blocks, stride, avg_down, dilation)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                conv3x3(in_channels, stem_channels // 2, stride=2),
                nn.BatchNorm2d(stem_channels // 2),
                nn.ReLU(inplace=True),
                conv3x3(stem_channels // 2, stem_channels // 2),
                nn.BatchNorm2d(stem_channels // 2),
                nn.ReLU(inplace=True),
                conv3x3(stem_channels // 2, stem_channels),
                nn.BatchNorm2d(stem_channels),
                nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = 'bn1', nn.BatchNorm2d(stem_channels)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, batch_dict):
        """Forward function."""
        x = batch_dict['camera_imgs']
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        
        batch_dict['image_features'] = tuple(outs)  
        return batch_dict

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

