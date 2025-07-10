import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# Variants:
# - no pooling
# - predict perturbation instead of new image
# - perturbation added at beginning instead of conditioned 

# Use feature-wise linear modulation to add the conditioned universal perturbation into the encoding of the face image with an affine transformation
class FiLM(nn.Module):
    def __init__(self, mask_encoder, in_channels):
        super().__init__()
        self.mask_encoder = mask_encoder
        self.pool = nn.AdaptiveAvgPool2d((1, 1)) # use this to make sure the feature map can be a 1D vector for modulation
        
        encoder_out_channels = self._get_encoder_output_dim()

        self.gamma_fc = nn.Linear(encoder_out_channels, in_channels)
        self.beta_fc = nn.Linear(encoder_out_channels, in_channels)
    
    def _get_encoder_output_dim(self):
        dummy_arr = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            feats = self.mask_encoder(dummy_arr)
        return feats[-1].shape[1]

    def forward(self, x, mask):
        mask_features = self.mask_encoder(mask)
        bottleneck = mask_features[-1]
        pooled = self.pool(bottleneck).squeeze(-1).squeeze(-1)
        gamma = self.gamma_fc(pooled).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta_fc(pooled).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta

class ConditionedUnet(nn.Module):
    def __init__(self, mask_encoder_name="resnet18"):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name = "resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=3
        )
        self.mask_encoder = smp.encoders.get_encoder(
            mask_encoder_name, 
            in_channels=3, 
            depth=5,
            weights="imagenet"
        )
        self.film = FiLM(self.mask_encoder, in_channels=512) #R18 has 512, R34 has 512, R50 has 2048

    def forward(self, x, mask):
        features = self.unet.encoder(x)
        bottleneck = features[-1]

        #mask = torch.randn_like(mask).to("cuda")
        modulated = self.film(bottleneck, mask)
        features = features[:-1] + [modulated]

        x = self.unet.decoder(features)
        x = self.unet.segmentation_head(x)

        x = torch.tanh(x)
        
        return x

