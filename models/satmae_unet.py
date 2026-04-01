import torch
import torch.nn as nn
from .satmae_backbone import SatMAESegmentationBackbone

class SatMAESegmentor(nn.Module):
    def __init__(self, num_classes=7, in_channels=3, pretrained=True):
        """
        A U-Net style segmentor using SatMAE++ as the encoder backbone.
        As required by the project guidelines, it utilizes multi-scale features
        from the transformer-based backbone.
        """
        super().__init__()
        # 1. ENCODER: SatMAE++ Transformer
        self.backbone = SatMAESegmentationBackbone(in_channels=in_channels, pretrained=pretrained)
        
        # 2. DECODER: Spatial Upsampling
        # SatMAE++ features: [B, 768, 14, 14] for 224x224 input
        self.decoder = nn.Sequential(
            # First Upsample (14x14 -> 56x56)
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=4), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Second Upsample (56x56 -> 224x224)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # Extract features from SatMAE++
        features = self.backbone(x)
        # Apply decoder to generate class logits
        logits = self.decoder(features)
        return logits

if __name__ == "__main__":
    # Quick sanity check for the model
    model = SatMAESegmentor(num_classes=7)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape (Logits): {output.shape}") # Expected: [1, 7, 224, 224]
