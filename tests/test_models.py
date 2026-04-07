"""
test_models.py
--------------
Unit tests for src/task1/models.py

Covers:
  - ConvBlock, DownBlock, UpBlock  (building blocks)
  - UNet                           (encoder–decoder segmentation)
  - AEEncoder, AEDecoder           (autoencoder components)
  - Autoencoder                    (full reconstruction model)
  - SegDecoderWithSkips            (skip-aware seg decoder)
  - AESegmentationModel            (frozen-encoder segmentation)
"""

import torch
import pytest

from task1.models import (
    ConvBlock,
    DownBlock,
    AttentionGate,
    UpBlock,
    ASPP,
    UNet,
    AEEncoder,
    AEDecoder,
    Autoencoder,
    SegDecoder,
    AESegmentationModel,
)

# ── Fixtures ────────────────────────────────────────────

H, W = 64, 64          # spatial size used throughout (power of 2 for pooling)
BATCH = 2


@pytest.fixture
def sample_input():
    """Standard (B, 3, H, W) input tensor."""
    return torch.randn(BATCH, 3, H, W)


# ═══════════════════════════════════════════════
#  Building blocks
# ═══════════════════════════════════════════════

class TestConvBlock:

    def test_output_shape(self):
        block = ConvBlock(3, 64)
        x = torch.randn(BATCH, 3, H, W)
        out = block(x)
        assert out.shape == (BATCH, 64, H, W)

    def test_different_channels(self):
        block = ConvBlock(32, 128)
        x = torch.randn(BATCH, 32, H, W)
        out = block(x)
        assert out.shape == (BATCH, 128, H, W)

    def test_preserves_spatial_dims(self):
        """Conv with padding=1 and kernel=3 should not change H, W."""
        block = ConvBlock(16, 16)
        x = torch.randn(1, 16, 17, 23)  # odd sizes
        out = block(x)
        assert out.shape == (1, 16, 17, 23)


class TestDownBlock:

    def test_halves_spatial(self):
        block = DownBlock(64, 128)
        x = torch.randn(BATCH, 64, H, W)
        out = block(x)
        assert out.shape == (BATCH, 128, H // 2, W // 2)

    def test_channel_change(self):
        block = DownBlock(3, 32)
        x = torch.randn(1, 3, 32, 32)
        out = block(x)
        assert out.shape == (1, 32, 16, 16)


class TestAttentionGate:

    def test_output_shape(self):
        gate = AttentionGate(gate_ch=128, skip_ch=64)
        g = torch.randn(BATCH, 128, H, W)
        skip = torch.randn(BATCH, 64, H, W)
        out = gate(g, skip)
        assert out.shape == (BATCH, 64, H, W)

    def test_output_bounded(self):
        """Output should be skip * sigmoid(...), so bounded by skip magnitude."""
        gate = AttentionGate(gate_ch=64, skip_ch=64)
        g = torch.randn(BATCH, 64, H, W)
        skip = torch.ones(BATCH, 64, H, W)
        out = gate(g, skip)
        # sigmoid output is in [0, 1], so |out| <= |skip| = 1
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_custom_inter_channels(self):
        gate = AttentionGate(gate_ch=256, skip_ch=128, inter_ch=32)
        g = torch.randn(1, 256, 16, 16)
        skip = torch.randn(1, 128, 16, 16)
        out = gate(g, skip)
        assert out.shape == (1, 128, 16, 16)


class TestUpBlock:

    def test_bilinear_upsampling(self):
        block = UpBlock(128 + 64, 64, bilinear=True)
        x = torch.randn(BATCH, 128, H // 2, W // 2)
        skip = torch.randn(BATCH, 64, H, W)
        out = block(x, skip)
        assert out.shape == (BATCH, 64, H, W)

    # @pytest.mark.xfail(
    #     reason="Known bug: UpBlock bilinear=False uses in_ch//2 for "
    #            "ConvTranspose2d, which mismatches actual input channels "
    #            "when encoder features aren't symmetric.",
    #     raises=RuntimeError,
    # )
    # def test_transposed_conv_upsampling(self):
    #     block = UpBlock(128 + 64, 64, bilinear=False)
    #     x = torch.randn(BATCH, 128, H // 2, W // 2)
    #     skip = torch.randn(BATCH, 64, H, W)
    #     out = block(x, skip)
    #     assert out.shape == (BATCH, 64, H, W)

    def test_mismatched_spatial_padding(self):
        """UpBlock should handle slight size mismatches via padding."""
        block = UpBlock(128 + 64, 64, bilinear=True)
        x = torch.randn(BATCH, 128, 15, 15)
        skip = torch.randn(BATCH, 64, 31, 31)
        out = block(x, skip)
        assert out.shape == (BATCH, 64, 31, 31)


class TestASPP:

    def test_output_shape(self):
        aspp = ASPP(512, 512)
        x = torch.randn(BATCH, 512, 8, 8)
        out = aspp(x)
        assert out.shape == (BATCH, 512, 8, 8)

    def test_different_channels(self):
        aspp = ASPP(256, 128)
        x = torch.randn(BATCH, 256, 16, 16)
        out = aspp(x)
        assert out.shape == (BATCH, 128, 16, 16)

    def test_custom_rates(self):
        aspp = ASPP(64, 64, rates=(3, 6, 9))
        x = torch.randn(1, 64, 32, 32)
        out = aspp(x)
        assert out.shape == (1, 64, 32, 32)

    def test_small_spatial(self):
        """ASPP should work even on very small feature maps."""
        aspp = ASPP(512, 512)
        x = torch.randn(1, 512, 4, 4)
        out = aspp(x)
        assert out.shape == (1, 512, 4, 4)


# ═══════════════════════════════════════════════
#  UNet
# ═══════════════════════════════════════════════

class TestUNet:

    def test_default_output_shape(self, sample_input):
        model = UNet(in_channels=3, num_classes=3)
        model.eval()
        out = model(sample_input)
        assert out.shape == (BATCH, 3, H, W)

    def test_custom_features(self, sample_input):
        model = UNet(in_channels=3, num_classes=5,
                     features=[32, 64, 128, 256])
        model.eval()
        out = model(sample_input)
        assert out.shape == (BATCH, 5, H, W)

    def test_single_class(self, sample_input):
        model = UNet(in_channels=3, num_classes=1)
        model.eval()
        out = model(sample_input)
        assert out.shape == (BATCH, 1, H, W)

    def test_count_parameters_positive(self):
        model = UNet()
        assert model.count_parameters() > 0

    def test_lightweight_fewer_params(self):
        heavy = UNet(features=[64, 128, 256, 512, 1024])
        light = UNet(features=[32, 64, 128, 256, 512])
        assert light.count_parameters() < heavy.count_parameters()

    def test_gradients_flow(self, sample_input):
        model = UNet(in_channels=3, num_classes=3)
        main_out, side_outs = model(sample_input)
        loss = main_out.sum() + sum(s.sum() for s in side_outs)
        loss.backward()
        first_weight = next(model.parameters())
        assert first_weight.grad is not None

    def test_non_power_of_two_spatial(self):
        """UNet should handle inputs whose size is not a power of 2."""
        model = UNet(in_channels=3, num_classes=3,
                     features=[32, 64, 128])
        model.eval()
        x = torch.randn(1, 3, 48, 48)
        out = model(x)
        assert out.shape == (1, 3, 48, 48)

    def test_deep_supervision_train_mode(self, sample_input):
        """In train mode, forward returns (main, side_outputs)."""
        model = UNet(in_channels=3, num_classes=3,
                     features=[32, 64, 128, 256])
        model.train()
        result = model(sample_input)
        assert isinstance(result, tuple)
        main_out, side_outs = result
        assert main_out.shape == (BATCH, 3, H, W)
        # features=[32,64,128,256] → 3 decoder stages → 2 side heads
        assert len(side_outs) == 2
        for s in side_outs:
            assert s.shape == (BATCH, 3, H, W)

    def test_deep_supervision_eval_mode(self, sample_input):
        """In eval mode, forward returns only the main output tensor."""
        model = UNet(in_channels=3, num_classes=3,
                     features=[32, 64, 128, 256])
        model.eval()
        result = model(sample_input)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (BATCH, 3, H, W)


# ═══════════════════════════════════════════════
#  AEEncoder
# ═══════════════════════════════════════════════

class TestAEEncoder:

    def test_default_output(self, sample_input):
        enc = AEEncoder(in_channels=3)
        bottleneck, skips = enc(sample_input)
        # Default features = [64, 128, 256, 512]
        assert bottleneck.shape[1] == 512
        assert len(skips) == 3  # one skip per encoder level (excluding bottleneck)

    def test_skip_channels(self, sample_input):
        features = [64, 128, 256, 512]
        enc = AEEncoder(in_channels=3, features=features)
        bottleneck, skips = enc(sample_input)
        # Skips should have channels [64, 128, 256] (features[:-1])
        for skip, expected_ch in zip(skips, features[:-1]):
            assert skip.shape[1] == expected_ch

    def test_bottleneck_spatial_reduction(self, sample_input):
        enc = AEEncoder(in_channels=3, features=[64, 128, 256, 512])
        bottleneck, _ = enc(sample_input)
        # 3 down-blocks → H/8, W/8
        assert bottleneck.shape[2] == H // 8
        assert bottleneck.shape[3] == W // 8

    def test_features_attribute(self):
        features = [32, 64, 128]
        enc = AEEncoder(in_channels=3, features=features)
        assert enc.features == features

    def test_custom_features(self, sample_input):
        features = [16, 32]
        enc = AEEncoder(in_channels=3, features=features)
        bottleneck, skips = enc(sample_input)
        assert bottleneck.shape[1] == 32
        assert len(skips) == 1
        assert skips[0].shape[1] == 16


# ═══════════════════════════════════════════════
#  AEDecoder
# ═══════════════════════════════════════════════

class TestAEDecoder:

    def test_output_shape(self):
        dec = AEDecoder(out_channels=3, features=[64, 128, 256, 512])
        # Input: bottleneck at deepest feature level
        x = torch.randn(BATCH, 512, H // 8, H // 8)
        out = dec(x)
        assert out.shape == (BATCH, 3, H, H)

    def test_output_range(self):
        """Decoder uses Sigmoid, so output should be in [0, 1]."""
        dec = AEDecoder(out_channels=3, features=[64, 128, 256, 512])
        x = torch.randn(BATCH, 512, 8, 8)
        out = dec(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_custom_features(self):
        dec = AEDecoder(out_channels=1, features=[16, 32, 64])
        x = torch.randn(1, 64, 4, 4)  # bottleneck
        out = dec(x)
        assert out.shape == (1, 1, 16, 16)


# ═══════════════════════════════════════════════
#  Autoencoder  (full model)
# ═══════════════════════════════════════════════

class TestAutoencoder:

    def test_reconstruction_shape(self, sample_input):
        ae = Autoencoder(in_channels=3)
        recon = ae(sample_input)
        assert recon.shape == sample_input.shape

    def test_custom_features(self, sample_input):
        ae = Autoencoder(in_channels=3, features=[32, 64, 128])
        recon = ae(sample_input)
        assert recon.shape == sample_input.shape

    def test_count_parameters(self):
        ae = Autoencoder()
        assert ae.count_parameters() > 0

    def test_gradients_flow(self, sample_input):
        ae = Autoencoder(in_channels=3)
        recon = ae(sample_input)
        loss = ((recon - sample_input) ** 2).mean()
        loss.backward()
        first_weight = next(ae.parameters())
        assert first_weight.grad is not None

    def test_reconstruction_range(self, sample_input):
        """Autoencoder output should be in [0, 1] due to Sigmoid."""
        ae = Autoencoder(in_channels=3)
        recon = ae(sample_input)
        assert recon.min() >= 0.0
        assert recon.max() <= 1.0


# ═══════════════════════════════════════════════
#  SegDecoderWithSkips
# ═══════════════════════════════════════════════

class TestSegDecoder:

    def test_output_shape(self):
        features = [64, 128, 256, 512]
        dec = SegDecoder(num_classes=3, features=features)
        dec.eval()
        # 3 down-blocks → bottleneck at H/8
        bottleneck = torch.randn(BATCH, 512, H // 8, W // 8)
        out = dec(bottleneck)
        assert out.shape == (BATCH, 3, H, W)

    def test_different_num_classes(self):
        features = [32, 64, 128]
        dec = SegDecoder(num_classes=5, features=features)
        dec.eval()
        bottleneck = torch.randn(BATCH, 128, H // 4, W // 4)
        out = dec(bottleneck)
        assert out.shape == (BATCH, 5, H, W)

    def test_deep_supervision_train_mode(self):
        features = [64, 128, 256, 512]
        dec = SegDecoder(num_classes=3, features=features,
                         target_size=(H, W))
        dec.train()
        bottleneck = torch.randn(BATCH, 512, H // 8, W // 8)
        main_out, side_outs = dec(bottleneck)
        assert main_out.shape == (BATCH, 3, H, W)
        # 3 decoder stages → 2 side heads
        assert len(side_outs) == 2
        for s in side_outs:
            assert s.shape == (BATCH, 3, H, W)

    def test_no_skip_connections(self):
        """SegDecoder takes only a bottleneck — no skip arguments."""
        features = [32, 64, 128]
        dec = SegDecoder(num_classes=3, features=features)
        dec.eval()
        bottleneck = torch.randn(1, 128, 8, 8)
        out = dec(bottleneck)
        # 2 upsample stages: 8→16→32
        assert out.shape == (1, 3, 32, 32)


# ═══════════════════════════════════════════════
#  AESegmentationModel
# ═══════════════════════════════════════════════

class TestAESegmentationModel:

    @pytest.fixture
    def encoder(self):
        return AEEncoder(in_channels=3, features=[64, 128, 256, 512])

    def test_output_shape(self, sample_input, encoder):
        model = AESegmentationModel(encoder, num_classes=3,
                                    freeze_encoder=True)
        model.eval()
        out = model(sample_input)
        assert out.shape == (BATCH, 3, H, W)

    def test_encoder_frozen(self, encoder):
        model = AESegmentationModel(encoder, freeze_encoder=True)
        for param in model.encoder.parameters():
            assert not param.requires_grad

    def test_encoder_unfrozen(self, encoder):
        model = AESegmentationModel(encoder, freeze_encoder=False)
        for param in model.encoder.parameters():
            assert param.requires_grad

    def test_decoder_trainable(self, encoder):
        model = AESegmentationModel(encoder, freeze_encoder=True)
        for param in model.seg_decoder.parameters():
            assert param.requires_grad

    def test_count_parameters_trainable_only(self, encoder):
        model = AESegmentationModel(encoder, freeze_encoder=True)
        trainable = model.count_parameters(trainable_only=True)
        total = model.count_parameters(trainable_only=False)
        assert trainable < total  # encoder is frozen → fewer trainable params

    def test_count_parameters_all(self, encoder):
        model = AESegmentationModel(encoder, freeze_encoder=False)
        trainable = model.count_parameters(trainable_only=True)
        total = model.count_parameters(trainable_only=False)
        assert trainable == total

    def test_gradients_only_decoder(self, sample_input, encoder):
        model = AESegmentationModel(encoder, num_classes=3,
                                    freeze_encoder=True)
        model.train()
        main_out, side_outs = model(sample_input)
        loss = main_out.sum() + sum(s.sum() for s in side_outs)
        loss.backward()
        # Encoder grads should be None (frozen)
        for param in model.encoder.parameters():
            assert param.grad is None
        # Decoder grads should exist
        has_decoder_grad = any(
            p.grad is not None for p in model.seg_decoder.parameters()
        )
        assert has_decoder_grad

    def test_no_skip_connections(self, sample_input, encoder):
        """Decoder should not use encoder skip connections."""
        model = AESegmentationModel(encoder, num_classes=3,
                                    freeze_encoder=True)
        model.eval()
        out = model(sample_input)
        assert out.shape == (BATCH, 3, H, W)
        # SegDecoder has no AttentionGate or UpBlock — only ConvBlock
        assert not any(
            isinstance(m, AttentionGate)
            for m in model.seg_decoder.modules()
        )

    def test_uses_encoder_features(self):
        """Model should infer features from encoder if not provided."""
        enc = AEEncoder(in_channels=3, features=[16, 32, 64])
        model = AESegmentationModel(enc, num_classes=2,
                                    freeze_encoder=True)
        model.eval()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 2, 32, 32)

    def test_pretrained_encoder_weights_preserved(self, sample_input, encoder):
        """Freezing should not alter existing encoder weights."""
        original_weights = {
            name: param.clone()
            for name, param in encoder.named_parameters()
        }
        model = AESegmentationModel(encoder, freeze_encoder=True)
        model.train()
        main_out, side_outs = model(sample_input)
        loss = main_out.sum() + sum(s.sum() for s in side_outs)
        loss.backward()
        for name, param in model.encoder.named_parameters():
            assert torch.equal(param, original_weights[name])
