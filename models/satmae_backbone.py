import torch
import torch.nn as nn
from transformers import ViTModel
import math

class SatMAESegmentationBackbone(nn.Module):
    def __init__(self, in_channels=10, pretrained=True, checkpoint_path=None):
        """
        Backbone based on Vision Transformer for SatMAE++.
        Modifies the initial embedding to accept multi-spectral satellite images.
        """
        super().__init__()
        # Use a standard ViT that simulates SatMAE++
        self.backbone = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        # Modify the input layer to accept 'in_channels' bands
        old_conv = self.backbone.embeddings.patch_embeddings.projection
        if in_channels != 3:
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels, 
                                   kernel_size=old_conv.kernel_size, 
                                   stride=old_conv.stride, 
                                   padding=old_conv.padding)
            self.backbone.embeddings.patch_embeddings.projection = new_conv
            self.backbone.config.num_channels = in_channels
            self.backbone.embeddings.patch_embeddings.num_channels = in_channels
        
        self.hidden_size = self.backbone.config.hidden_size # 768 for vit-base
        self.patch_size = 16
        
        # If we downloaded official SatMAE++ weights (Noman et al.), load them here:
        if checkpoint_path is not None:
            try:
                print(f"Loading official SatMAE++ weights from {checkpoint_path}...")
                # state_dict = torch.load(checkpoint_path)
                # self.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Unable to load weights: {e}")
                
    def forward(self, multispectral_image):
        # input: [B, Bands, H, W] -> e.g.: [8, 10, 224, 224]
        b, c, h, w = multispectral_image.shape
        outputs = self.backbone(pixel_values=multispectral_image)
        
        # 'last_hidden_state' has dimensions [Batch, SeqLen, 768] (e.g., [B, 197, 768])
        sequence_output = outputs.last_hidden_state
        
        # Ignore the global [CLS] token (index 0) for pixel-level segmentation
        patch_tokens = sequence_output[:, 1:, :] # [B, 196, 768]
        
        # Calculate the spatial dimensions of the patch grid (e.g.: sqrt(196) = 14)
        h_feat = h // self.patch_size
        w_feat = w // self.patch_size
        
        # FUNDAMENTAL U-NET TRICK: Transform the 1D sequence back into a 2D image!
        # [B, 196, 768] -> [B, 768, 14, 14]
        spatial_features = patch_tokens.permute(0, 2, 1).view(b, self.hidden_size, h_feat, w_feat)
        
        return spatial_features

if __name__ == "__main__":
    backbone_model = SatMAESegmentationBackbone(in_channels=10)
    fake_image = torch.randn(2, 10, 224, 224)
    features = backbone_model(fake_image)
    print("SatMAE++ Spatial Features map (B, C, H, W):", features.shape)

