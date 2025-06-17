import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- basic blocks ---------- #
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1)
        )
    def forward(self, x):
        return x + self.body(x)


class CBAM(nn.Module):
    def __init__(self, ch, k=7, r=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch // r, 1, bias=False), nn.ReLU(True),
            nn.Conv2d(ch // r, ch, 1, bias=False)
        )
        self.spatial = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        # channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        att = self.mlp(avg_pool) + self.mlp(max_pool)
        x = x * torch.sigmoid(att)
        # spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        x = x * torch.sigmoid(self.spatial(torch.cat([avg, mx], dim=1)))
        return x


def down(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, 2, 1), nn.ReLU(True),
        ResBlock(out_c), ResBlock(out_c)
    )


class FusionUNet(nn.Module):
    def __init__(self, in_ch=3, base=64, use_blur=True):
        super().__init__()
        self.use_blur = use_blur and in_ch == 3
        branch = 3 if self.use_blur else 2

        # encoder branches
        self.a1 = down(1, base)
        self.a2 = down(base, base * 2)
        self.a3 = down(base * 2, base * 4)

        self.b1 = down(1, base)
        self.b2 = down(base, base * 2)
        self.b3 = down(base * 2, base * 4)

        if self.use_blur:
            self.bl1 = down(1, base)
            self.bl2 = down(base, base * 2)
            self.bl3 = down(base * 2, base * 4)

        # fusion & bottleneck
        self.cbam = CBAM(base * 4 * branch)
        self.bottle = nn.Conv2d(base * 4 * branch, base * 4, 1)

        # decoder stage 3
        self.red3 = nn.Conv2d(base * 4 + base * 2 * branch, base * 4, 1)
        self.dec3 = ResBlock(base * 4)
        # decoder stage 2
        self.red2 = nn.Conv2d(base * 4 + base * branch, base * 2, 1)
        self.dec2 = ResBlock(base * 2)
        # decoder stage 1
        in_ch_red1 = base * 2 + (1 if self.use_blur else 0)
        self.red1 = nn.Conv2d(in_ch_red1, base, 1)
        self.head = nn.Sequential(
            ResBlock(base),
            nn.Conv2d(base, 1, 3, 1, 1)
        )

    @staticmethod
    def _up(x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        # split inputs
        tfi = x[:, 0:1]
        tfp = x[:, 1:2]
        blur = (x[:, 2:3] if self.use_blur else None)

        # encoder A
        a1 = self.a1(tfi)
        a2 = self.a2(a1)
        a3 = self.a3(a2)
        # encoder B
        b1 = self.b1(tfp)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        # optional blur branch
        if self.use_blur:
            bl1 = self.bl1(blur)
            bl2 = self.bl2(bl1)
            bl3 = self.bl3(bl2)
            fused = torch.cat([a3, b3, bl3], dim=1)
        else:
            fused = torch.cat([a3, b3], dim=1)

        # fusion + bottleneck
        f = self.bottle(self.cbam(fused))

        # decode level 3
        up3 = self._up(f)
        cat3 = torch.cat([up3, a2, b2] + ([bl2] if self.use_blur else []), dim=1)
        d3 = self.dec3(self.red3(cat3))
        # decode level 2
        up2 = self._up(d3)
        cat2 = torch.cat([up2, a1, b1] + ([bl1] if self.use_blur else []), dim=1)
        d2 = self.dec2(self.red2(cat2))
        # decode level 1 + head
        up1 = self._up(d2)
        if self.use_blur:
            blur_up = F.interpolate(blur, size=up1.shape[-2:], mode='bilinear', align_corners=False)
            up1 = torch.cat([up1, blur_up], dim=1)
        out = torch.sigmoid(self.head(self.red1(up1)))
        return out

# example usage:
# net = FusionUNet(in_ch=3, base=64, use_blur=True)
# pred = net(torch.randn(2, 3, 256, 256))
