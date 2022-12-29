import argparse
from typing import Tuple
import torch.nn.functional as F
from torch import nn, Tensor
from util.misc import *
from .position_embedding import build_position_encoding


class Convolutional(nn.Module):
    def __init__(self, img_channel, filters, size, stride):
        # type: (int, int, int, int) -> None
        super(Convolutional, self).__init__()
        self.conv = nn.Conv2d(in_channels=img_channel, out_channels=filters, kernel_size=(size, size),
                              stride=(stride, stride), padding=size // 2, bias=False)
        self.bn = nn.BatchNorm2d(num_features=filters)

    def forward(self, inputs):
        # type: (Tensor) -> Tensor
        outputs = self.conv(inputs)
        outputs = self.bn(outputs)
        return F.leaky_relu(outputs)


class Res_block(nn.Module):
    def __init__(self, img_channel):
        # type: (int) -> None
        super(Res_block, self).__init__()
        self.conv_1 = Convolutional(img_channel, img_channel // 2, 1, 1)
        self.conv_3 = Convolutional(img_channel // 2, img_channel, 3, 1)

    def forward(self, inputs):
        # type: (Tensor) -> Tensor
        outputs = self.conv_1(inputs)
        outputs = self.conv_3(outputs)
        return inputs + outputs


class DarkNet(nn.Module):
    def __init__(self, args):
        # type: (argparse.Namespace) -> None
        super(DarkNet, self).__init__()
        self.img_size = args.img_size
        self.channels = 32
        self.cov_1 = Convolutional(img_channel=3, filters=self.channels, size=3, stride=1)
        self.cov_2 = Convolutional(img_channel=self.channels, filters=self.channels * 2, size=3, stride=2)
        self.channels *= 2
        self.img_size /= 2
        self.res_net = []
        self.num_res_blocks = [1, 2, 8, 8, 4]
        for i, res_block in enumerate(self.num_res_blocks):
            self.res_net.extend([Res_block(img_channel=self.channels) for _ in range(res_block)])
            if i != len(self.num_res_blocks) - 1:
                self.res_net.append(Convolutional(img_channel=self.channels, filters=self.channels * 2, size=3,
                                                  stride=2))
                self.channels *= 2
                self.img_size /= 2
        self.res_net = nn.Sequential(*self.res_net)
        self.max_pool_5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=(5 - 1) // 2)
        self.max_pool_9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=(9 - 1) // 2)
        self.max_pool_13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=(13 - 1) // 2)
        self.channels *= 4

    def forward(self, inputs):
        # type: (Tensor) -> Tensor
        outputs = inputs
        outputs = self.cov_1(outputs)
        outputs = self.cov_2(outputs)
        outputs = self.res_net(outputs)

        outputs_5 = self.max_pool_5(outputs)
        outputs_9 = self.max_pool_9(outputs)
        outputs_13 = self.max_pool_13(outputs)
        outputs = torch.cat([outputs, outputs_5, outputs_9, outputs_13], dim=1)
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
        pos = self[1](xs).to(xs.tensors.dtype)

        return out, pos


def build_backbone(args, hyp):
    position_embedding = build_position_encoding(args, hyp)
    backbone = DarkNet(args)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.channels
    return model
