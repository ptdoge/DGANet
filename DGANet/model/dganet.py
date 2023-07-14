import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import resnet
from ._blocks import Conv1x1, Conv3x3, get_norm_layer
from ._utils import KaimingInitMixin, Identity


class Backbone(nn.Module, KaimingInitMixin):
    def __init__(self, in_ch, arch, pretrained=True, strides=(2, 1, 2, 2, 2)):
        super().__init__()

        if arch == 'resnet18':
            self.resnet = resnet.resnet18(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet34':
            self.resnet = resnet.resnet34(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        elif arch == 'resnet50':
            self.resnet = resnet.resnet50(pretrained=pretrained, strides=strides, norm_layer=get_norm_layer())
        else:
            raise ValueError

        self._trim_resnet()

        if in_ch != 3:
            self.resnet.conv1 = nn.Conv2d(
                in_ch,
                64,
                kernel_size=7,
                stride=strides[0],
                padding=3,
                bias=False
            )

        if not pretrained:
            self._init_weight()

    def forward(self, x):
        # x 3 256 256
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        return x1, x2, x3, x4

    def _trim_resnet(self):
        self.resnet.avgpool = Identity()
        self.resnet.fc = Identity()


# DGAM
class CoAttention(nn.Module):
    def __init__(self, in_ch, ratio=4):
        super().__init__()
        # T1, T2, Abs-Diff, SA-T1, SA-T2

        self.conv_query = nn.Conv2d(in_channels=in_ch, out_channels=in_ch // ratio, kernel_size=1, bias=False)

        self.conv_key = nn.Conv2d(in_channels=in_ch, out_channels=in_ch // ratio, kernel_size=1, bias=False)

        self.conv_value = Conv1x1(in_ch, in_ch, norm=True, act=True)

        self.conv_sa = nn.Conv2d(in_channels=in_ch, out_channels=1, kernel_size=3, stride=1, padding=1)

        self.conv_out = Conv1x1(2 * in_ch, in_ch, norm=True, act=True)

    def forward(self, t):
        b, _, h, w = t.size()

        # 
        x = self.conv_sa(t)
        x1, x2 = torch.split(x, b // 2, dim=0)
        x = F.softmax(torch.cat([x1, x2], dim=1), dim=1)
        x1, x2 = torch.split(x, 1, dim=1)  # b//2 1 h w

        # 
        t1, t2 = torch.split(t, b // 2, dim=0)
        tdiff = torch.abs(t1 - t2)
        query = self.conv_query(tdiff).view(b // 2, -1, h * w).permute(0, 2, 1)  # b//2 (h*w) c
        k = self.conv_key(t).view(b, -1, h * w)
        v = self.conv_value(t).view(b, -1, h * w)
        k1, k2 = torch.split(k, b // 2, dim=0)
        v1, v2 = torch.split(v, b // 2, dim=0)

        k = torch.cat([k1, k2], dim=-1)  # b//2 c 2*(h*w)
        v = torch.cat([v1, v2], dim=-1)

        w_matric = torch.matmul(query, k)  # b//2 (h*w) 2*(h*w)
        w_matric = F.softmax(w_matric, dim=-1).permute(0, 2, 1)

        v = torch.matmul(v, w_matric).view(b // 2, -1, h, w)

        # 
        t1 = torch.cat([t1, x1 * v], dim=1)
        t2 = torch.cat([t2, x2 * v], dim=1)
        t = torch.cat([t1, t2], dim=0)
        t = self.conv_out(t)

        return t


# STAM
class SelfAttention(nn.Module):  #
    def __init__(self, in_ch, ratio=4):
        super().__init__()
        self.conv_query = nn.Conv2d(in_channels=in_ch, out_channels=in_ch // ratio, kernel_size=1, bias=False)

        self.conv_key = nn.Conv2d(in_channels=in_ch, out_channels=in_ch // ratio, kernel_size=1, bias=False)

        self.conv_value = Conv1x1(in_ch, in_ch, norm=True, act=True)

        self.conv_out = Conv1x1(2 * in_ch, in_ch, norm=True, act=True)

    def forward(self, t):
        b, _, h, w = t.size()

        t1, t2 = torch.split(t, b // 2, dim=0)
        tcat = torch.cat([t1, t2], dim=-1)  # h (2*w)
        query = self.conv_query(tcat).view(b // 2, -1, h * 2 * w).permute(0, 2, 1)  # b//2 (h*2*w) c

        k = self.conv_key(tcat).view(b // 2, -1, h * 2 * w)
        v = self.conv_value(tcat).view(b // 2, -1, h * 2 * w)

        w_matric = torch.matmul(query, k)  # b//2 (h*2*w) (h*2*w)
        w_matric = F.softmax(w_matric, dim=1)

        v = torch.matmul(v, w_matric).view(b // 2, -1, h, 2 * w)
        v1, v2 = torch.split(v, w, dim=-1)
        t1 = torch.cat([t1, v1], dim=1)
        t2 = torch.cat([t2, v2], dim=1)
        t = torch.cat([t1, t2], dim=0)
        t = self.conv_out(t)

        return t


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio, kernel_size):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        ca = self.ca(x)
        sa = self.sa(x)
        return torch.sqrt(torch.clamp(ca * sa, min=1e-12))


class Decoder(nn.Module, KaimingInitMixin):
    def __init__(self, fc_ch):
        super().__init__()
        self.dr1 = Conv1x1(64, 96, norm=True, act=True)
        self.dr2 = Conv1x1(128, 96, norm=True, act=True)
        self.dr3 = Conv1x1(256, 96, norm=True, act=True)
        self.dr4 = Conv1x1(512, 96, norm=True, act=True)
        self.conv_out = nn.Sequential(
            Conv3x3(384, 256, norm=True, act=True),
            nn.Dropout(0.5),
            Conv1x1(256, fc_ch, norm=True, act=True)
        )

        self.coa = CoAttention(in_ch=96)

        self._init_weight()

    def forward(self, feats):
        f1 = self.dr1(feats[0])  #
        f2 = self.dr2(feats[1])  #
        f3 = self.dr3(feats[2])  #
        f4 = self.dr4(feats[3])  #

        f4 = self.coa(f4)

        f2 = F.interpolate(f2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(f3, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(f4, size=f1.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([f1, f2, f3, f4], dim=1)  #
        y = self.conv_out(x)  #

        return y


class Base(nn.Module):
    def __init__(self, in_ch, fc_ch=64):
        super().__init__()
        self.extract = Backbone(in_ch=in_ch, arch='resnet18')  #
        self.decoder = Decoder(fc_ch=fc_ch)
        #
        self.cbam = CBAM(fc_ch, ratio=8, kernel_size=7)
        self.conv = Conv3x3(128, 64, norm=True, act=True)

    def forward(self, t1, t2):
        b, _, _, _ = t1.size()
        t = torch.cat([t1, t2], dim=0)
        f = self.extract(t)
        f = self.decoder(f)

        f_1, f_2 = torch.split(f, b, dim=0)

        weight = self.cbam(self.conv(torch.cat([f_1, f_2], dim=1)))
        f_1 = weight * f_1
        f_2 = weight * f_2

        dist = torch.norm(f_1 - f_2, dim=1, keepdim=True)  # 1 64 64
        dist = F.interpolate(dist, size=t1.shape[2:], mode='bilinear', align_corners=True)

        return dist, dist


if __name__ == '__main__':
    model = Base(in_ch=3, fc_ch=64)
    t1 = torch.randn(1, 3, 64, 64)
    t2 = torch.randn(1, 3, 64, 64)
    dist, _ = model(t1, t2)
    print(dist.size())
