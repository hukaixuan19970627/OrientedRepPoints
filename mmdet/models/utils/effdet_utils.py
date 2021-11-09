import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmdet.ops import ConvModule


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 norm_cfg=dict(type='BN', momentum=0.003, eps=1e-4, requires_grad=True),
                 activation=None,
                 bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=False)
        self.pointwise = ConvModule(in_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=None, bias=bias, inplace=False)
        if activation == "ReLU":
            self.act = nn.ReLU()
        elif activation == "Swish":
            self.act = MemoryEfficientSwish()
        else:
            self.act = None

    def init_weights(self):
        xavier_init(self.depthwise, distribution='uniform')
        xavier_init(self.pointwise.conv, distribution='uniform')

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.act:
            x = self.act(x)
        return x
