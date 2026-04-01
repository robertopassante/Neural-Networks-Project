import torch
import torch.nn as nn

class CLIPTextQuerySegmentation(nn.Module):
    """
    Advanced module for Text-Prompted Segmentation using CLIP embeddings.
    Allows the model to segment regions based on text prompts like 'Find roads'.
    """
    def __init__(self, c_img=768, c_text=512):
        super().__init__()
        # 1. CLIP and projectors (stubs)
        self.text_proj = nn.Linear(c_text, c_img)
        self.mask_generator = nn.Conv2d(c_img, 1, kernel_size=1) 

    def forward(self, image_features, text_emb):
        """
        Conditional multiplication (Image Features * Text Query)
        image_features: [B, C, H, W] from SatMAE++
        text_emb: [B, 512] from CLIP Text Encoder
        """
        text_emb_proj = self.text_proj(text_emb) # [B, 768]
        text_emb_spatial = text_emb_proj.unsqueeze(-1).unsqueeze(-1) # [B, 768, 1, 1]
        
        conditioned_features = image_features * text_emb_spatial
        segmentation_mask = torch.sigmoid(self.mask_generator(conditioned_features))
        return segmentation_mask
