import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class ConvBlock(nn.Module):
    """Residual block: two conv -> BN -> ReLU layers with a shortcut connection.

    When in_ch != out_ch a 1x1 convolution projects the input to match,
    otherwise the shortcut is an identity mapping.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # 1x1 projection shortcut when channel dimensions differ
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class DownBlock(nn.Module):
    """Max‑pool → ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsample -> concatenate skip -> ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                                  align_corners=True)
            self.conv = ConvBlock(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2,
                                         kernel_size=2, stride=2)
            self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if sizes don't exactly match
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2,
                       dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling.

    Applies parallel dilated convolutions at multiple rates plus global
    average pooling, then fuses the results with a 1x1 projection.
    Placed at the encoder bottleneck to capture multi-scale context.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 rates: tuple = (6, 12, 18)):
        super().__init__()

        # 1x1 convolution branch
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # Dilated convolution branches
        self.atrous_convs = nn.ModuleList()
        for rate in rates:
            self.atrous_convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=rate,
                          dilation=rate, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ))

        # Global average pooling branch (no BN after pooling to 1x1
        # to avoid BatchNorm errors with single-element spatial dims)
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.gap_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=True),
            nn.ReLU(inplace=True),
        )

        # Fuse all branches: 1x1 + len(rates) dilated + GAP
        num_branches = 1 + len(rates) + 1
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * num_branches, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        branches = [self.conv1x1(x)]
        for atrous in self.atrous_convs:
            branches.append(atrous(x))
        # GAP branch — upsample back to input spatial size
        gap = self.gap_conv(self.gap_pool(x))
        gap = F.interpolate(gap, size=(h, w), mode="bilinear",
                            align_corners=True)
        branches.append(gap)

        return self.project(torch.cat(branches, dim=1))


# ═══════════════════════════════════════════════
#  1.  UNet
# ═══════════════════════════════════════════════

class UNet(nn.Module):
    """
    Standard U-Net for multi-class segmentation.

    Architecture
    ------------
    Encoder:  [64, 128, 256, 512, 1024]  (configurable via `features`)
    Decoder:  mirrors the encoder with skip connections

    Parameters count depends on `features`.
        features=[64,128,256,512,1024] -> ~31 M params (default)
        features=[32,64,128,256,512]   -> ~7.8 M params (lightweight)
    """

    def __init__(self,
                 in_channels: int = config.IMAGE_CHANNELS,
                 num_classes: int = config.NUM_CLASSES,
                 features: list = None,
                 bilinear: bool = True):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512, 1024]

        self.inc = ConvBlock(in_channels, features[0])

        # Encoder (downsampling)
        self.encoders = nn.ModuleList()
        for i in range(len(features) - 1):
            self.encoders.append(DownBlock(features[i], features[i + 1]))

        # ASPP at the bottleneck for multi-scale context
        self.aspp = ASPP(features[-1], features[-1])

        # Decoder (upsampling)
        self.decoders = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.decoders.append(
                UpBlock(features[i] + features[i - 1], features[i - 1],
                        bilinear=bilinear)
            )

        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        skips = []
        x = self.inc(x)
        skips.append(x)

        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # Remove bottleneck from skips (it's our starting point for decoder)
        skips = skips[:-1]

        # Multi-scale context at the bottleneck
        x = self.aspp(x)

        # Decoder path
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.outc(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AEEncoder(nn.Module):
    """
    Encoder half of the autoencoder.
    Downsamples the image through conv blocks + max-pooling, producing
    multi-scale feature maps (returned for skip connections later).
    """

    def __init__(self,
                 in_channels: int = config.IMAGE_CHANNELS,
                 features: list = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.inc = ConvBlock(in_channels, features[0])

        self.encoders = nn.ModuleList()
        for i in range(len(features) - 1):
            self.encoders.append(DownBlock(features[i], features[i + 1]))

        self.features = features

    def forward(self, x):
        """Returns bottleneck and list of skip features (high‑res first)."""
        skips = []
        x = self.inc(x)
        skips.append(x)

        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        bottleneck = skips.pop()   # deepest feature map
        return bottleneck, skips


class AEDecoder(nn.Module):
    """
    Decoder for image reconstruction (autoencoder pre‑training).
    Mirrors the encoder, using transposed convolutions to upsample.
    No skip connections during AE training.
    """

    def __init__(self,
                 out_channels: int = config.IMAGE_CHANNELS,
                 features: list = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        reversed_feats = list(reversed(features))

        self.ups = nn.ModuleList()
        for i in range(len(reversed_feats) - 1):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(reversed_feats[i], reversed_feats[i + 1],
                                   kernel_size=2, stride=2),
                ConvBlock(reversed_feats[i + 1], reversed_feats[i + 1]),
            ))

        self.final = nn.Sequential(
            nn.Conv2d(reversed_feats[-1], out_channels, kernel_size=1),
            nn.Sigmoid(),  # pixel values in [0, 1]
        )

    def forward(self, x):
        for up in self.ups:
            x = up(x)
        return self.final(x)


class Autoencoder(nn.Module):
    """
    Full autoencoder: encoder + reconstruction decoder.
    Trained to minimise reconstruction loss (MSE) on raw images.
    """

    def __init__(self,
                 in_channels: int = config.IMAGE_CHANNELS,
                 features: list = None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.encoder = AEEncoder(in_channels, features)
        self.decoder = AEDecoder(in_channels, features)

    def forward(self, x):
        bottleneck, _skips = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return reconstruction

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════
#  3.  Segmentation decoder on frozen AE encoder
# ═══════════════════════════════════════════════

class SegDecoderWithSkips(nn.Module):
    """
    A segmentation decoder that takes the bottleneck and skip features
    from a (frozen) AEEncoder and produces a class mask.

    This uses skip connections from the encoder, similar to a U-Net decoder,
    to recover spatial detail for segmentation.
    """

    def __init__(self,
                 num_classes: int = config.NUM_CLASSES,
                 features: list = None,
                 bilinear: bool = True):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        # Decoder mirrors encoder in reverse
        self.ups = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.ups.append(
                UpBlock(features[i] + features[i - 1], features[i - 1],
                        bilinear=bilinear)
            )

        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, bottleneck, skips):
        """
        Parameters
        ----------
        bottleneck : tensor from deepest encoder stage
        skips      : list of tensors, high-res first  [skip0, skip1, ...]
        """
        x = bottleneck
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self.outc(x)


class AESegmentationModel(nn.Module):
    """
    Combines a frozen AEEncoder with a trainable SegDecoderWithSkips.

    Usage
    -----
    1. Train an Autoencoder (see train_autoencoder.py).
    2. Load the trained encoder weights into this model.
    3. Freeze the encoder and train only the segmentation decoder.
    """

    def __init__(self,
                 encoder: AEEncoder,
                 num_classes: int = config.NUM_CLASSES,
                 features: list = None,
                 freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if features is None:
            features = encoder.features

        self.seg_decoder = SegDecoderWithSkips(num_classes, features)

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        return self.seg_decoder(bottleneck, skips)

    def count_parameters(self, trainable_only: bool = True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters()
                       if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    device = "cpu"
    x = torch.randn(2, 3, 256, 256, device=device)

    # UNet
    unet = UNet().to(device)
    out = unet(x)
    print(f"UNet  input: {x.shape}  ->  output: {out.shape}")
    print(f"UNet  trainable params: {unet.count_parameters():,}")

    # Autoencoder
    ae = Autoencoder().to(device)
    recon = ae(x)
    print(f"\nAutoencoder  input: {x.shape}  ->  reconstruction: {recon.shape}")
    print(f"Autoencoder  trainable params: {ae.count_parameters():,}")

    # AE-based segmentation
    ae_seg = AESegmentationModel(ae.encoder, freeze_encoder=True).to(device)
    seg = ae_seg(x)
    print(f"\nAE-Seg  input: {x.shape}  ->  output: {seg.shape}")
    print(f"AE-Seg  trainable params (decoder only): "
          f"{ae_seg.count_parameters(True):,}")
    print(f"AE-Seg  total params: "
          f"{ae_seg.count_parameters(False):,}")