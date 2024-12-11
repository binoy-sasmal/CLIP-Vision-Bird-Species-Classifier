import torch
from torch import nn
import clip

class CLIPClassifier(nn.Module):
    def __init__(self, num_classes=25, encoder_type="ViT-B/32"):
        super(CLIPClassifier, self).__init__()
        self.clip_model, _ = clip.load(encoder_type)
        self.fc1 = nn.Linear(self.clip_model.visual.output_dim, 256)
        self.dropout = nn.Dropout(0.5)  # Regularization to prevent overfitting
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        with torch.no_grad():  # Freeze CLIP encoder
            x = self.clip_model.encode_image(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.fc2(x)
