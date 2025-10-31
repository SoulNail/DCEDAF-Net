import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# -----------------------------------#
#   定义一个简单的单输入网络
# -----------------------------------#
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=False),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()

        # 第一条路径: 1x1 卷积
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 第二条路径: 3x3 卷积
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # 第三条路径: 3x3 卷积 + 3x3 卷积
        self.branch3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 最后再经过 1x1 卷积
        self.final_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        # 第一条路径输出
        branch1_out = self.branch1(x)

        # 第二条路径输出
        branch2_out = self.branch2(x)

        # 第三条路径输出
        branch3_out = self.branch3_1(x)
        branch3_out = self.branch3_2(branch3_out)

        # 拼接三个路径的输出 (沿通道维度拼接)
        concat_out = torch.cat([branch1_out, branch2_out, branch3_out], dim=1)

        # 通过最终的 1x1 卷积
        output = self.final_conv(concat_out)

        return output


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class ECAAttention(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        y=self.gap(x)  # 在空间方向执行全局平均池化: (B,C,H,W)-->(B,C,1,1)
        y=y.squeeze(-1).permute(0,2,1)  # 将通道描述符去掉一维,便于在通道上执行卷积操作:(B,C,1,1)-->(B,C,1)-->(B,1,C)
        y=self.conv(y)  # 在通道维度上执行1D卷积操作,建模局部通道之间的相关性: (B,1,C)-->(B,1,C)
        y=self.sigmoid(y) # 生成权重表示: (B,1,C)
        y=y.permute(0,2,1).unsqueeze(-1)  # 重塑shape: (B,1,C)-->(B,C,1)-->(B,C,1,1)
        return x*y.expand_as(x)  # 权重对输入的通道进行重新加权: (B,C,H,W) * (B,C,1,1) = (B,C,H,W)
        # output = x * y.expand_as(x)
        # return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:(B,C,H,W)
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 最大值和对应的索引.
        avg_result = torch.mean(x, dim=1, keepdim=True)  # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 平均值
        result = torch.cat([max_result, avg_result], 1)  # 在通道上拼接两个矩阵:(B,2,H,W)
        output = self.conv(result)  # 然后重新降维为1维:(B,1,H,W)
        output = self.sigmoid(output)  # 通过sigmoid获得权重:(B,1,H,W)
        return output


class ECASABlock(nn.Module):

    def __init__(self, channel=512, reduction=16, kernel_size=49):
        super().__init__()
        self.ECAAttention = ECAAttention(kernel_size=kernel_size)
        self.SpatialAttention = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # (B,C,H,W)
        B, C, H, W = x.size()
        residual1 = x
        out = x * self.ECAAttention(x)  # 将输入与通道注意力权重相乘: (B,C,H,W) * (B,C,1,1) = (B,C,H,W)
        residual2 = x  # 新加的
        out = out * self.SpatialAttention(out)  # 将更新后的输入与空间注意力权重相乘:(B,C,H,W) * (B,1,H,W) = (B,C,H,W)
        return out + residual1 + residual2


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # 使用 max(1, channels // reduction) 确保输出通道数大于 0
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)

    def forward(self, x):
        # Squeeze
        batch_size, num_channels, _, _ = x.size()
        # 使用 reshape 替代 view
        y = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, num_channels)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).reshape(batch_size, num_channels, 1, 1)
        return x * y


class ResidualAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ResidualAttentionModule, self).__init__()

        # Main branch: 3x3 Conv -> BatchNorm -> ReLU -> 3x3 Conv -> BatchNorm -> ReLU
        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),  # 禁用 inplace
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False)  # 禁用 inplace
        )

        # Residual connection: 1x1 Conv
        self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # Squeeze-and-Excitation (SE) block
        self.se_block = SEBlock(in_channels)

    def forward(self, x):
        # Main branch
        out = self.conv_branch(x)

        # Residual connection
        residual = self.residual_conv(x)

        # Add main and residual branches without inplace modification
        out = out + residual

        # Apply SE block
        out = self.se_block(out)

        return out


class IPUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super(IPUNet, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Conv1 = InceptionBlock(img_ch, 64)
        self.Conv2 = InceptionBlock(64, 128)
        self.Conv3 = InceptionBlock(128, 256)
        self.Conv4 = InceptionBlock(256, 512)
        self.Conv5 = InceptionBlock(512, 1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        self.ECASABlock4 = ECASABlock(channel=512, reduction=16, kernel_size=7)
        self.ECASABlock3 = ECASABlock(channel=256, reduction=16, kernel_size=7)
        self.ECASABlock2 = ECASABlock(channel=128, reduction=16, kernel_size=7)
        self.ECASABlock1 = ECASABlock(channel=64, reduction=16, kernel_size=7)
        self.RAM5 = ResidualAttentionModule(1024)  ####新加的
        self.RAM4 = ResidualAttentionModule(512)  ####新加的
        self.RAM3 = ResidualAttentionModule(256)  ####新加的
        self.RAM2 = ResidualAttentionModule(128)  ####新加的

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)#Conv有inception修改

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        # x4 = self.ECASABlock4(x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.RAM5(d5)####新加的
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        # x3 = self.ECASABlock3(x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.RAM4(d4)  ####新加的
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        # x2 = self.ECASABlock2(x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.RAM3(d3)  ####新加的
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        # x1 = self.ECASABlock1(x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.RAM2(d2)  ####新加的
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = F.softmax(d1,dim=1)  # mine

        return d1

# if __name__ == '__main__':
#     model = MyNet(1, 2)
#     from torchsummary.torchsummary import summary
#     summary(model, (1, 64, 64), batch_size=1, device='cpu')
