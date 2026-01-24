import functools
import jittor as jt
from jittor import nn
import numpy as np
from collections import OrderedDict

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block)
    return nn.Sequential(*layers)

class Module_with_Init(nn.Module):
    def __init__(self,):
        super().__init__()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = jt.randn(m.weight.data.shape) * 0.02
                if m.bias is not None:
                    m.bias.data = jt.randn(m.bias.data.shape) * 0.02
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data = jt.randn(m.weight.data.shape) * 0.02

    def lrelu(self, x):
        outt = jt.maximum(0.2*x, x)
        return outt

class ResConvBlock_CBAM(nn.Module):
    def __init__(self, in_nc, nf=64, res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.cbam = CBAM(nf)
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.relu(self.conv1(x))
        out = self.res_scale * self.cbam(self.relu(self.conv2(x))) + x
        return x + out * self.res_scale

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        nf (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
    """

    def __init__(self, nf=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.ReLU()

    def execute(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class conv1x1(nn.Module):
    def __init__(self, in_nc, out_nc, is_activate=True):
        super().__init__()
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=1, padding=0, stride=1)
        self.is_activate = is_activate
        if is_activate:
            self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        if self.is_activate:
            x = self.relu(x)
        return x

class ResidualBlock3D(nn.Module):
    def __init__(self, in_c, out_c, is_activate=True):
        super().__init__()
        self.activation = nn.ReLU() if is_activate else nn.Identity()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, stride=1)

        if in_c != out_c:
            self.short_cut = nn.Conv3d(in_c, out_c, kernel_size=1, padding=0, stride=1)
        else:
            self.short_cut = None

    def execute(self, x):
        output = self.conv1(x)
        output = self.activation(output)
        output = self.conv2(output)
        
        if self.short_cut is not None:
            output += self.short_cut(x)
        else:
            output += x
        output = self.activation(output)
        return output

class conv3x3(nn.Module):
    def __init__(self, in_nc, out_nc, stride=2, is_activate=True):
        super().__init__()
        self.conv = nn.Conv2d(in_nc, out_nc, kernel_size=3, padding=1, stride=stride)
        self.is_activate = is_activate
        if is_activate:
            self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        if self.is_activate:
            x = self.relu(x)
        return x

class convWithBN(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, stride=1, is_activate=True, is_bn=True):
        super(convWithBN, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding,
                              stride=stride, bias=False)
        self.is_bn = is_bn
        self.is_activate = is_activate
        if is_bn:
            self.bn = nn.BatchNorm2d(out_c)
        if is_activate:
            self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        if self.is_bn:
            x = self.bn(x)
        if self.is_activate:
            x = self.relu(x)
        return x


class DoubleCvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleCvBlock, self).__init__()
        self.conv1 = convWithBN(in_c, out_c, kernel_size=3, padding=1, stride=1, is_bn=False)
        self.conv2 = convWithBN(out_c, out_c, kernel_size=3, padding=1, stride=1, is_bn=False)

    def execute(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        return output

class nResBlocks(nn.Module):
    def __init__(self, nf, nlayers=2):
        super().__init__()
        self.blocks = make_layer(ResidualBlock(nf, nf), n_layers=nlayers)
    
    def execute(self, x):
        return self.blocks(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, is_activate=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = convWithBN(in_c, out_c, kernel_size=3, padding=1, stride=1, is_bn=False)
        self.conv2 = convWithBN(out_c, out_c, kernel_size=3, padding=1, stride=1, is_activate=False, is_bn=False)

        if in_c != out_c:
            self.short_cut = convWithBN(in_c, out_c, kernel_size=1, padding=0, stride=1, is_activate=False, is_bn=False)
        else:
            self.short_cut = None
        
        self.activation = nn.LeakyReLU(0.2) if is_activate else nn.Identity()

    def execute(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.activation(output)
        
        if self.short_cut is not None:
            output += self.short_cut(x)
        else:
            output += x
        return output

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.in_nc = in_planes
        self.ratio = ratio
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        avgout = self.avg_pool(x)
        avgout = self.conv1(avgout)
        avgout = self.relu(avgout)
        avgout = self.conv2(avgout)
        
        maxout = self.max_pool(x)
        maxout = self.conv1(maxout)
        maxout = self.relu(maxout)
        maxout = self.conv2(maxout)
        
        return self.sigmoid(avgout + maxout)
    

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def execute(self, x):
        avgout = jt.mean(x, dim=1, keepdims=True)
        maxout = jt.max(x, dim=1, keepdims=True)
        x = jt.concat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
    
    def execute(self, x):
        x = self.ca(x) * x
        out = self.sa(x) * x
        return out

class MaskMul(nn.Module):
    def __init__(self, scale_factor=1):
        super().__init__()
        self.scale_factor = scale_factor

    def execute(self, x, mask):
        if mask.shape[1] != x.shape[1]:
            mask = jt.mean(mask, dim=1, keepdims=True)
        pooled_mask = nn.avg_pool2d(mask, self.scale_factor)
        out = x * pooled_mask
        return out

class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, out_channels=None, up_scale=2, mode='bilinear'):
        super(UpsampleBLock, self).__init__()
        self.mode = mode
        self.up_scale = up_scale
        
        if mode == 'pixel_shuffle':
            self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
            self.up = nn.PixelShuffle(up_scale)
        elif mode == 'bilinear':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            print(f"Please tell me what is '{mode}' mode ????")
            raise NotImplementedError
        self.relu = nn.ReLU()

    def execute(self, x):
        x = self.conv(x)
        if self.mode == 'pixel_shuffle':
            x = self.up(x)
        else:
            # Jittor使用resize进行上采样，需要计算目标尺寸
            h, w = x.shape[2], x.shape[3]
            target_h, target_w = h * self.up_scale, w * self.up_scale
            x = nn.resize(x, size=(target_h, target_w), mode='bilinear')
        x = self.relu(x)
        return x

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = jt.zeros((downscale_factor * downscale_factor * c,
                       1, downscale_factor, downscale_factor))
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    
    return nn.conv2d(input, kernel, stride=downscale_factor, groups=c)

class PixelUnshuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor
    
    def execute(self, input):
        '''
        input: batchSize * c * k*w * k*h
        kdownscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return pixel_unshuffle(input, self.downscale_factor)

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    
    def padding(self, tensors):
        if len(tensors) > 2: 
            return tensors
        x, y = tensors
        xb, xc, xh, xw = x.shape
        yb, yc, yh, yw = y.shape
        diffY = xh - yh
        diffX = xw - yw
        y = nn.pad(y, (diffX // 2, diffX - diffX//2, 
                       diffY // 2, diffY - diffY//2))
        return (x, y)

    def execute(self, x, dim=None):
        x = self.padding(x)
        return jt.concat(x, dim=dim if dim is not None else self.dim)

if __name__ == '__main__':
    jt.flags.use_cuda = 1  # 使用 GPU
    
    x = jt.randn((1, 32, 16, 16))
    
    # 测试各个模块
    print("Testing Jittor modules conversion...")
    
    # 测试 CBAM
    cbam = CBAM(32)
    out = cbam(x)
    print(f"CBAM output shape: {out.shape}")
    
    # 测试 ResidualBlock
    res_block = ResidualBlock(32, 32)
    out = res_block(x)
    print(f"ResidualBlock output shape: {out.shape}")
    
    # 测试 UpsampleBlock
    up = UpsampleBLock(32, 32, up_scale=2, mode='bilinear')
    out = up(x)
    print(f"UpsampleBlock output shape: {out.shape}")
    
    # 测试 Concat
    concat = Concat()
    y = jt.randn((1, 32, 14, 14))
    out = concat([x, y])
    print(f"Concat output shape: {out.shape}")
    
    print("All tests passed!")