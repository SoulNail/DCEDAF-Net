#
# lr_x = lr_x.double()
# lr_y = lr_y.double()
# hr_x = hr_x.double()
# l_a = l_a.double()
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from models.Additional.ELA import EfficientLocalizationAttention
from models.Additional.PConv import Pinwheel_shapedConv
from models.Additional.RFAConv import RFCBAMConv
from models.Additional.DWT.wad_module import wad_module



class DropBlock(nn.Module):
    def __init__(self, block_size=5, p=0.1):
        super().__init__()
        self.block_size = block_size
        self.p = p

    def calculate_gamma(self, x):
        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid

    def forward(self, x):
        N, C, H, W = x.size()
        if self.training:
            gamma = self.calculate_gamma(x)
            mask_shape = (N, C, H - self.block_size + 1, W - self.block_size + 1)
            mask = torch.bernoulli(torch.full(mask_shape, gamma, device=x.device))
            mask = F.pad(mask, [self.block_size // 2] * 4, value=0)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x


class SelfAttentionBlock(nn.Module):
    def __init__(self):
        super(SelfAttentionBlock, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)
        x3 = torch.cat((x1, x2), dim=1)
        x4 = torch.sigmoid(self.conv(x3))
        x = x4 * x
        assert len(x.shape) == 4
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttentionBlock, self).__init__()

        self.inter_channels = in_channels
        self.in_channels = in_channels
        self.gating_channels = in_channels

        self.theta = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            nn.BatchNorm2d(self.inter_channels),
        )
        self.phi = nn.Sequential(
            nn.Conv2d(in_channels=self.gating_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.inter_channels),
            nn.Sigmoid(),
        )

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode='bilinear', align_corners=True)
        f = F.relu(theta_x + phi_g, inplace=True)

        psi_f = self.psi(f)

        return psi_f


class LinearAttention(nn.Module):
    """线性注意力机制 - 复杂度为O(HW)而不是O(HW²)"""

    def __init__(self, dim):
        super(LinearAttention, self).__init__()
        self.dim = dim
        self.reduction_ratio = 8
        self.q_conv = nn.Conv2d(dim, dim // self.reduction_ratio, kernel_size=1)
        self.k_conv = nn.Conv2d(dim, dim // self.reduction_ratio, kernel_size=1)
        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        batch_size, c, h, w = x.size()

        # 生成Q, K, V投影
        q = self.q_conv(x).view(batch_size, -1, h * w)  # B x C' x N
        k = self.k_conv(y).view(batch_size, -1, h * w)  # B x C' x N
        v = self.v_conv(y).view(batch_size, -1, h * w)  # B x C x N

        # 应用softmax到K上 (在空间维度上)
        k = self.softmax(k)  # B x C' x N

        # 计算上下文向量
        context = torch.bmm(v, k.permute(0, 2, 1))  # B x C x C'

        # 应用上下文向量到Q上
        out = torch.bmm(context, q)  # B x C x N

        # 重塑并应用残差连接
        out = out.view(batch_size, c, h, w)
        out = self.gamma * out + x

        return out




class M_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(M_Conv, self).__init__()
        # pad_size = kernel_size // 2
        # self.conv = nn.Sequential(
        #     nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=pad_size, stride=1),
        #     nn.BatchNorm2d(output_channels),
        #     nn.ReLU(inplace=True),
        # )
        # self.conv = RFAConv(input_channels, output_channels, 3, 1)
        self.conv = Pinwheel_shapedConv(input_channels, output_channels, 3, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class N_Conv(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(N_Conv, self).__init__()
        pad_size = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, padding=pad_size, stride=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        x = self.conv(x)
        return x


def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output


def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class ConvNext(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()
        pad_size = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad_size, groups=dim)  # depthwise conv
        self.norm = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim * 4, kernel_size=1, stride=1, padding=0)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(dim * 4, dim, kernel_size=1, stride=1, padding=0)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_block = DropBlock(7, 0.5)

    def forward(self, x):
        _input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.gamma.unsqueeze(-1).unsqueeze(-1) * x
        x = _input + self.drop_block(x)

        return x


class SkipConnection(nn.Module):
    def __init__(self, dim):
        super(SkipConnection, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(dim * 2, dim, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
        self.RFCBAMConv = RFCBAMConv(in_channel=dim, out_channel=dim, kernel_size=3)

    def forward(self, x, x_s):
        # 假设 x 的形状为 (batch, channels, H, W)
        x_avg_pooled = self.avg_pool(x)  # 平均池化
        x1 = self.conv1(x_avg_pooled)
        x_max_pooled = self.max_pool(x)  # 最大池化
        x2 = self.conv2(x_max_pooled)

        combined = torch.cat([x1, x2], dim=1)  # 在通道维度拼接

        x = self.conv3(combined)
        x = self.sigmoid(x)

        # 这里加一点对x_s处理的东西，比如：RFAConv
        x_s = self.RFCBAMConv(x_s)


        output = torch.cat([x, x_s], dim=1)

        return output


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class DCEDAFNet(nn.Module):
    def __init__(self, channel, n_classes, base_c, depths, kernel_size):
        super(DCEDAFNet, self).__init__()

        self.input_layer = nn.Sequential(
            M_Conv(channel, base_c * 1, kernel_size=3),
            *[ConvNext(base_c * 1, kernel_size) for _ in range(depths[0])]
        )

        self.input_skip = nn.Sequential(
            M_Conv(channel, base_c * 1, kernel_size=3),
        )
        self.conv1 = M_Conv(channel, base_c * 1, kernel_size=3)

        self.skip1 = SkipConnection(base_c * 1)

        self.down_conv_2 = nn.Sequential(*[
            nn.Conv2d(base_c * 2, base_c * 2, kernel_size=2, stride=2),
            # wad_module(),
            *[ConvNext(base_c * 2, kernel_size) for _ in range(depths[1])]
            ])
        self.conv2 = M_Conv(channel, base_c * 2, kernel_size=3)

        self.skip2 = SkipConnection(base_c * 2)

        self.down_conv_3 = nn.Sequential(*[
            nn.Conv2d(base_c * 4, base_c * 4, kernel_size=2, stride=2),
            # wad_module(),
            *[ConvNext(base_c * 4, kernel_size) for _ in range(depths[2])]
            ])
        self.conv3 = M_Conv(channel, base_c * 4, kernel_size=3)

        self.skip3 = SkipConnection(base_c * 4)

        self.down_conv_4 = nn.Sequential(*[
            nn.Conv2d(base_c * 8, base_c * 8, kernel_size=2, stride=2),
            # wad_module(),
            *[ConvNext(base_c * 8, kernel_size) for _ in range(depths[3])]
            ])
        self.attn = EfficientLocalizationAttention(base_c * 8)
        # self.attn = SelfAttentionBlock()

        self.up_residual_conv3 = ResidualConv(base_c * 8, base_c * 4, 1, 1)
        self.up_residual_conv2 = ResidualConv(base_c * 4, base_c * 2, 1, 1)
        self.up_residual_conv1 = ResidualConv(base_c * 2, base_c * 1, 1, 1)

        self.output_layer3 = nn.Sequential(
            nn.Conv2d(base_c * 4, n_classes, 1, 1),
            nn.Sigmoid(),
        )
        self.output_layer2 = nn.Sequential(
            nn.Conv2d(base_c * 2, n_classes, 1, 1),
            nn.Sigmoid(),
        )
        self.output_layer1 = nn.Sequential(
            nn.Conv2d(base_c * 1, n_classes, 1, 1),
            nn.Sigmoid(),
        )

        # self.fgf = FastGuidedFilter_attention(r=2, eps=1e-2)

        self.attention_block3 = CrossAttentionBlock(base_c * 8)
        self.attention_block2 = CrossAttentionBlock(base_c * 4)
        self.attention_block1 = CrossAttentionBlock(base_c * 2)

        self.conv_cat_3 = N_Conv(base_c * 8 + base_c * 8, base_c * 8, kernel_size=1)
        self.conv_cat_2 = N_Conv(base_c * 8 + base_c * 4, base_c * 4, kernel_size=1)
        self.conv_cat_1 = N_Conv(base_c * 4 + base_c * 2, base_c * 2, kernel_size=1)

    def forward(self, x):
        # Get multi-scale from input
        _, _, h, w = x.size()

        x_scale_2 = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        x_scale_3 = F.interpolate(x, size=(h // 4, w // 4), mode='bilinear', align_corners=True)


        # Encoder
        x1 = self.input_layer(x) + self.input_skip(x)
        x1_conv = self.conv1(x)
        x1_down = torch.cat([x1_conv, x1], dim=1)

        x2 = self.down_conv_2(x1_down)
        x2_conv = self.conv2(x_scale_2)
        x2_down = torch.cat([x2_conv, x2], dim=1)

        x3 = self.down_conv_3(x2_down)
        x3_conv = self.conv3(x_scale_3)
        x3_down = torch.cat([x3_conv, x3], dim=1)

        x4 = self.down_conv_4(x3_down)
        x4 = self.attn(x4)


        # skip
        x1_skip = self.skip1(x1, x1_conv)
        x2_skip = self.skip2(x2, x2_conv)
        x3_skip = self.skip3(x3, x3_conv)

        x1_down = x1_down + x1_skip
        x2_down = x2_down + x2_skip
        x3_down = x3_down + x3_skip

        # Decoder
        """加finalcross就是取消最后两行注释"""
        _, _, h, w = x3_down.size()
        x4_up = F.interpolate(x4, size=(h, w), mode='bilinear', align_corners=True)     # 上一层特征上采样
        x3_gf = torch.cat([x3_down, x4_up], dim=1)
        x3_gf_conv = self.conv_cat_3(x3_gf)             # 主干网络的特征
        x3_cross = self.attention_block3(x3_gf_conv, x4_up)     # 主干网络的特征和上一层的特征做交叉注意力
        x3_conv = self.up_residual_conv3(x3_cross)      # 注意力增强后的特征

        """加finalcross就是取消最后两行注释"""
        _, _, h, w = x2_down.size()
        x3_conv = F.interpolate(x3_conv, size=(h, w), mode='bilinear', align_corners=True)
        x3_up = F.interpolate(x3_gf_conv, size=(h, w), mode='bilinear', align_corners=True)
        x2_gf = torch.cat([x2_down, x3_up], dim=1)
        x2_gf_conv = self.conv_cat_2(x2_gf)
        x2_cross = self.attention_block2(x2_gf_conv, x3_conv)
        x2_conv = self.up_residual_conv2(x2_cross)


        """加finalcross就是取消最后两行注释"""
        _, _, h, w = x1_down.size()
        x2_conv = F.interpolate(x2_conv, size=(h, w), mode='bilinear', align_corners=True)
        x2_up = F.interpolate(x2_gf_conv, size=(h, w), mode='bilinear', align_corners=True)
        x1_gf = torch.cat([x1_down, x2_up], dim=1)
        x1_gf_conv = self.conv_cat_1(x1_gf)
        x1_cross = self.attention_block1(x1_gf_conv, x2_conv)
        x1_conv = self.up_residual_conv1(x1_cross)

        """finalcross"""
        _, _, h, w = x.size()
        x3_out = F.interpolate(x3_conv, size=(h, w), mode='bilinear', align_corners=True)
        out_3 = self.output_layer3(x3_out)
        out_2 = self.output_layer2(x2_conv)
        out_1 = self.output_layer1(x1_conv)

        """nofinalcross消融的部分，修改了这部分还要修改output_layer123，把第一个base_dim加倍"""
        # out_3 = F.interpolate(x3_gf_conv, size=(h, w), mode='bilinear', align_corners=True)
        # out_2 = F.interpolate(x2_gf_conv, size=(h, w), mode='bilinear', align_corners=True)
        # out_3 = self.output_layer3(out_3)
        # out_2 = self.output_layer2(out_2)
        # out_1 = self.output_layer1(x1_gf_conv)

        return out_1, out_2, out_3

# if __name__ == '__main__':
#     # from torchsummary import summary
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     in_channels = 3
#     n_classes = 1
#     depths = [3, 3, 9, 3]
#     base_c = 64
#     kernel_size = 3
#     model = DCEDAFNet(in_channels, n_classes, base_c, depths=depths, kernel_size=kernel_size, device=device).to(device)
#
#     input_data = (torch.randn(1, 3, 608, 608)).to(device)
#     output = model(input_data)
