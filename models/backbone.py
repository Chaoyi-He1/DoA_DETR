import argparse
from typing import Tuple
import torch.nn.functional as F
from torch import nn, Tensor
from util.misc import *
from .position_embedding import build_position_encoding


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

        
class Convolutional(nn.Module):
    def __init__(self, img_channel, filters, size, stride):
        # type: (int, int, int, int) -> None
        super(Convolutional, self).__init__()
        self.conv = nn.Conv2d(in_channels=img_channel, out_channels=filters, kernel_size=(size, size),
                              stride=(stride, stride), padding=size // 2, bias=True)
        self.bn = nn.BatchNorm2d(num_features=filters)

    def forward(self, inputs):
        # type: (Tensor) -> Tensor
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        return F.relu(outputs, inplace=True)


class Res_block(nn.Module):
    def __init__(self, img_channel, drop_path_ratio):
        # type: (int, float) -> None
        super(Res_block, self).__init__()
        self.conv_1 = Convolutional(img_channel, img_channel // 2, 1, 1)
        self.conv_3 = Convolutional(img_channel // 2, img_channel, 3, 1)
        self.drop_path = DropPath(drop_prob=drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()

    def forward(self, inputs):
        # type: (Tensor) -> Tensor
        outputs = self.conv_1(inputs)
        outputs = self.conv_3(outputs)
        outputs = self.drop_path(outputs)
        return inputs + outputs


class DarkNet(nn.Module):
    def __init__(self, args):
        # type: (argparse.Namespace) -> None
        super(DarkNet, self).__init__()
        self.img_size = args.img_size
        self.channels = 32
        self.cov_1 = Convolutional(img_channel=args.img_channel, filters=self.channels, size=3, stride=1)
        self.cov_2 = Convolutional(img_channel=self.channels, filters=self.channels * 2, size=3, stride=2)
        self.channels *= 2
        self.img_size /= 2
        self.res_net = nn.ModuleList()
        num_res_blocks = [1, 2, 4, 4, 2]
        for i, res_block in enumerate(num_res_blocks):
            self.res_net.extend([Res_block(img_channel=self.channels, 
                                           drop_path_ratio=args.drop_path_ratio) for _ in range(res_block)])
            if i != len(num_res_blocks) - 1:
                self.res_net.append(Convolutional(img_channel=self.channels, filters=self.channels * 2, size=3,
                                                  stride=2))
                self.channels *= 2
                self.img_size /= 2
        # self.res_net = nn.Sequential(*self.res_net)
        self.spp = nn.ModuleList([nn.MaxPool2d(kernel_size=3, stride=1, padding=(3 - 1) // 2),
                                  nn.MaxPool2d(kernel_size=5, stride=1, padding=(5 - 1) // 2), 
                                  nn.MaxPool2d(kernel_size=9, stride=1, padding=(9 - 1) // 2), 
                                  nn.MaxPool2d(kernel_size=13, stride=1, padding=(13 - 1) // 2)])
        self.channels *= 5

    def forward(self, inputs):
        # type: (Tensor) -> Tensor
        outputs = inputs
        outputs = self.cov_1(outputs)
        outputs = self.cov_2(outputs)
        for layer in self.res_net:
            outputs = layer(outputs)

        # outputs_5 = self.max_pool_5(outputs)
        # outputs_9 = self.max_pool_9(outputs)
        # outputs_13 = self.max_pool_13(outputs)
        spp_out = [outputs] 
        for max_pool in self.spp:
            spp_out.append(max_pool(outputs))
        outputs = torch.cat(spp_out, dim=1)
        return outputs


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        # type: (NestedTensor) -> Tuple[NestedTensor, Tensor]
        xs = self[0](tensor_list.tensors)
        m = tensor_list.mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=xs.shape[-2:]).to(torch.bool)[0]
        out = NestedTensor(xs, mask)
        # position encoding
        pos = self[1](out).to(out.tensors.dtype)

        return out, pos


def build_backbone(args, hyp):
    position_embedding = build_position_encoding(args, hyp)
    backbone = DarkNet(args)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.channels
    return model
