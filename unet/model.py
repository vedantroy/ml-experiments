import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, c_in: int, c_mid: int, c_out: int):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        k = 3
        # padding is better but I'm trying to follow the paper pretty strictly
        # https://github.com/milesial/Pytorch-UNet/issues/68
        c1 = nn.Conv2d(c_in, c_mid, kernel_size=k)
        assert c1.weight.shape == (c_mid, c_in, k, k)
        c2 = nn.Conv2d(c_mid, c_out, kernel_size=k)
        assert c2.weight.shape == (c_out, c_mid, k, k)
        self.conv = nn.Sequential(
            c1,
            nn.ReLU(inplace=True),
            c2,
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch_size, in_channels, W, H = x.shape
        assert in_channels == self.c_in
        y = self.conv(x)
        assert y.shape == (batch_size, self.c_out, W - 4, H - 4)
        return y


class Down(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, level: int):
        super().__init__()
        # For debugging
        self.level = level
        double_conv = DoubleConv(in_channels, mid_channels, out_channels)
        # We start w/ the pool b/c we need the output of the double_conv
        # for the skip connections
        pool = nn.MaxPool2d(2)
        self.down = nn.Sequential(pool, double_conv)

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, c_in: int, c_mid: int, c_out: int, level: int):
        super().__init__()
        # For debugging
        self.level = level
        self.c_in = c_in
        after_up_conv = c_in / 2
        assert after_up_conv.is_integer()
        after_up_conv = int(after_up_conv)
        self.after_up_conv = after_up_conv

        # Since the downwards path ends w/ a conv
        # We should start the upwards path with a upsample/transposed convolution
        # The `stride=2` is what causes the up-scaling of the image
        self.up = nn.ConvTranspose2d(c_in, after_up_conv, kernel_size=2, stride=2)
        assert self.up.weight.shape == (c_in, after_up_conv, 2, 2)
        self.conv = DoubleConv(c_in, c_mid, c_out)

    def forward(self, x, from_skip_connection):
        _, _, before_upscale_W, before_upscale_H = x.shape
        upscaled = self.up(x)
        B, C, W, H = upscaled.shape
        assert C == self.after_up_conv
        assert W == before_upscale_W * 2
        assert H == before_upscale_H * 2

        _, prevC, prevW, prevH = from_skip_connection.shape
        assert prevC + C == self.c_in
        assert prevW > W
        assert prevH > H
        diff_W = (prevW - W) // 2
        diff_H = (prevH - H) // 2

        # The original image is too big since convolutions downsize the width/height
        # The alternative is to pad the output with zeroes, this prevents loss of information
        # which is theoretically better (practice seems to show no difference?)
        cropped = from_skip_connection[:, :, diff_W : diff_W + W, diff_H : diff_H + H]

        concatted = torch.cat([cropped, upscaled], dim=1)
        assert concatted.shape == (B, self.c_in, W, H)
        return self.conv(concatted)


class UNet(nn.Module):
    def __init__(self, n_in_channels, n_classes):
        super().__init__()
        self.in_conv = DoubleConv(n_in_channels, 64, 64)
        self.down1 = Down(64, 128, 128, 1)
        self.down2 = Down(128, 256, 256, 2)
        self.down3 = Down(256, 512, 512, 3)
        self.down4 = Down(512, 1024, 1024, 4)
        self.up1 = Up(1024, 512, 512, 4)
        self.up2 = Up(512, 256, 256, 3)
        self.up3 = Up(256, 128, 128, 2)
        self.up4 = Up(128, 64, 64, 1)
        self.classifier = nn.Conv2d(64, n_classes, kernel_size=1)
        assert self.classifier.weight.shape == (n_classes, 64, 1, 1)

    def forward(self, x):
        _, _, W, H = x.shape
        x_out = self.in_conv(x)
        x1_out = self.down1(x_out)
        x2_out = self.down2(x1_out)
        x3_out = self.down3(x2_out)
        x4_out = self.down4(x3_out)

        x = self.up1(x4_out, x3_out)
        x = self.up2(x, x2_out)
        x = self.up3(x, x1_out)
        x = self.up4(x, x_out)

        x =  self.classifier(x)
        _ ,_, finalW, finalH = x.shape

        # In the original UNet paper, the image goes from (572, 572) => (388, 388)
        # 572 - 388 = 184
        # My net decreases each side by 188 (I suspect there is an extra convolution in here?)
        assert (184 <= W - finalW <= 188) and (184 <= H - finalH <= 188)
        return x