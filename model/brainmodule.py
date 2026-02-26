# brain_module.py
# Implementation of the Brain Module from "ImageDecodingMEG"
# Based on Section 2.3 and Appendix A

import torch
from torch import nn
import typing as tp

# Importing components from the provided common.py
from .common import (
    ChannelMerger,
    SubjectLayers,
    ConvSequence
)

class MLPProjector(nn.Module):
    """
    MLP Projector Head as described in Appendix A [2060].
    Consists of Layer Norm -> GELU -> Linear blocks.
    
    Based on Table S1, the projector maps 2048 -> 768 features.
    The parameter count (~1.57M) suggests a single dense layer 
    wrapped in normalization and activation.
    """
    def __init__(self, in_dim: int, out_dim: int, num_layers: int = 1):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.LayerNorm(in_dim, elementwise_affine=False))
            layers.append(nn.GELU())
            layers.append(nn.Linear(in_dim, out_dim))
            # If multiple layers were used, we would update in_dim here
            # but for the specific architecture in Table S1, it's a direct mapping.
            in_dim = out_dim 
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BrainModule(nn.Module):
    """
    The MEG Decoding Model f_theta described in ImageDecodingMEG.
    
    Flow:
    1. Spatial Attention (ChannelMerger) [2049]
    2. Subject-specific Linear Layer (SubjectLayers) [2050]
    3. Dilated ConvNet Backbone (ConvSequence) [2051]
    4. Feature Projection to F' (Linear/Conv1x1) [2054]
    5. Temporal Aggregation (Learned Affine) [2057]
    6. Projection Heads (CLIP/MSE) [2060]
    """
    def __init__(
        self,
        # Dimensions
        n_time_steps: int = 181,    # T: Default based on Table S1 (1.5s @ 120Hz approx)
        in_channels: int = 272,     # C: MEG channels
        n_subjects: int = 4,        # S: Number of subjects
        
        # Architecture Hyperparams (Table S1)
        spatial_out: int = 270,     # Channels after spatial attention
        backbone_channels: tp.List[int] = [270, 320, 320, 320, 320], # Input -> Block1 -> Block2
        feature_dim: int = 2048,    # F': Intermediate feature dimension
        target_dim: int = 768,      # F: Output dimension (e.g., CLIP-Vision CLS)
        
        # Conv Backbone Specs [2052]
        kernel_size: int = 3,
        dilation_growth: int = 2,   # Paper implies standard dilation growth handled internally or fixed
        
        # Heads
        use_clip_head: bool = True,
        use_mse_head: bool = True,
    ):
        super().__init__()
        
        # 1. Spatial Attention [2049] & [2150]
        # Maps raw sensors C to spatial_out. 
        # Uses ChannelMerger from common.py which implements the Fourier spatial attention.
        self.spatial_attention = ChannelMerger(
            chout=spatial_out,
            pos_dim=2048, # 2048 is valid for FourierEmb (32*32*2)
            n_subjects=n_subjects,
            per_subject=False # Shared spatial attention as per SpeechDecodingMEG/ImageDecodingMEG
        )

        # 1.5. Mixing Layer (User Request)
        # 1x1 Conv without activation between Spatial Attention and Subject Layer
        self.mixing_layer = nn.Conv1d(spatial_out, spatial_out, 1)

        # 2. Subject Specific Layer [2050] & [2150]
        # 1x1 Conv unique to each subject.
        self.subject_layer = SubjectLayers(
            in_channels=spatial_out,
            out_channels=spatial_out,
            n_subjects=n_subjects,
            init_id=True
        )

        # 3. Convolutional Backbone [2051] - [2053]
        # "Succession of 1D convolutional blocks... residual skip connections"
        # Table S1 lists 2 blocks. 
        # We use ConvSequence from common.py.
        # Note: common.py's ConvSequence handles the residual/activation logic.
        self.backbone = ConvSequence(
            channels=backbone_channels,
            kernel=kernel_size,
            stride=1,
            dilation_growth=dilation_growth,
            skip=True,         # Residual connections
            dropout_input=0.1,
            dropout=0.5,
            activation=nn.GELU, # [2053] uses GELU for first two layers
            glu=2,             # [2053] "last one use a GLU activation" -> GLU every 3rd layer
            glu_context=1,
            glu_glu=True,     # Standard GLU behavior
            batch_norm=True,   # Standard in the baseline
        )
        
        # 4. Linear Projection to F' [2054]
        # "Output... passed through a learned linear projection to yield F' features"
        last_conv_dim = backbone_channels[-1] # 320

        self.feature_projection = nn.Sequential(
            nn.Conv1d(last_conv_dim, last_conv_dim * 2, 1),
            nn.GELU(),
            nn.Conv1d(last_conv_dim * 2, feature_dim, 1) # feature_dim = 2048
        )

        # 5. Temporal Aggregation [2057]
        # "Learned affine projection... projected from R^T to R using learned weight w and bias b"
        # We implement this as a Linear layer mapping T -> 1.
        self.temporal_aggregator = nn.Linear(n_time_steps, 1, bias=True)

        # 6. Projection Heads [2060]
        self.heads = nn.ModuleDict()
        
        if use_clip_head:
            self.heads['clip'] = MLPProjector(feature_dim, target_dim)
            
        if use_mse_head:
            # Often the MSE head might target a different dim depending on the latent 
            # (e.g. 768 for CLS, higher for others), but defaulting to target_dim here.
            self.heads['mse'] = MLPProjector(feature_dim, target_dim)

    def forward(self, x: torch.Tensor, subject_indices: torch.Tensor):
        """
        Args:
            x: MEG Data (Batch, Sensors, Time) -> (B, 272, 181)
            subject_indices: Subject IDs (Batch,)
        """
        
        # 1. Spatial Attention
        # common.py ChannelMerger expects a 'batch' object usually, 
        # but here we mock the interface or rely on simple arguments if possible.
        # However, ChannelMerger in common.py strictly requires a batch object 
        # with .subject_index and .meg attributes or helper methods.
        # To make this standalone compatible with the provided common.py logic:
        
        class MockBatch:
            def __init__(self, s, meg):
                self.subject_index = s
                self.meg = meg
                # Mocking internal list needed by PositionGetter in common.py
                # In a real scenario, this connects to the MNE info.
                # Assuming PositionGetter handles the layout internally or is pre-cached.
                self._recordings = [None] * len(s) 
            
            def __len__(self):
                return len(self.subject_index) 

        mock_batch = MockBatch(subject_indices, x)
        
        # Note: ChannelMerger.forward calls PositionGetter. 
        # If running without actual MNE data linkage, PositionGetter in common.py 
        # might fail if not initialized with layouts. 
        # Assuming the environment is set up correctly as per SpeechDecodingMEG.
        x = self.spatial_attention(x, mock_batch) # (B, 270, T)

        # 1.5. Mixing Layer
        x = self.mixing_layer(x)

        # 2. Subject Layer
        x = self.subject_layer(x, subject_indices) # (B, 270, T)

        # 3. Backbone
        x = self.backbone(x) # (B, 320, T)

        # 4. Feature Projection
        x = self.feature_projection(x) # (B, 2048, T)

        # 5. Temporal Aggregation
        # [2057] Learned affine projection from R^T to R
        # Input to Linear needs to be (B, Features, Time) -> (B, Features, 1)
        x = self.temporal_aggregator(x) 
        x = x.squeeze(-1) # (B, 2048)

        # 6. Heads
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(x) # (B, 768)

        return outputs