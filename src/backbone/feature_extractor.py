import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image


class FeatureExtractor:
    def __init__(self, device=None):
        # Set up device
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load pretrained ResNet-18
        weights = ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)

        # Freeze all parameters — no gradient updates, this backbone is fixed
        for param in self.model.parameters():
            param.requires_grad = False

        # Set to eval mode (affects BatchNorm behavior)
        self.model.eval()
        self.model.to(self.device)

        # Dictionary to store captured feature maps
        self._features = {}

        # Register forward hooks on layer2 and layer3
        self.model.layer2.register_forward_hook(self._make_hook("layer2"))
        self.model.layer3.register_forward_hook(self._make_hook("layer3"))

        # Preprocessing pipeline
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _make_hook(self, layer_name):
        def hook(module, input, output):
            self._features[layer_name] = output.detach()
        return hook

    def get_combined_features(self, image_path):
        """
        Returns a single combined feature map by upsampling layer3
        to match layer2's spatial resolution, then concatenating
        channel-wise.
        Output shape: [C_total, H, W]  (no batch dimension)
        """
        features = self.extract_features(image_path)
        layer2 = features["layer2"]  # [1, 128, 32, 32]
        layer3 = features["layer3"]  # [1, 256, 16, 16]

        # Upsample layer3 to match layer2's spatial size
        layer3_upsampled = torch.nn.functional.interpolate(
            layer3, size=layer2.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate along channel dimension
        combined = torch.cat([layer2, layer3_upsampled], dim=1)  # [1, 384, 32, 32]

        return combined.squeeze(0)  # [384, 32, 32]


    def extract_features(self, image_path):
        """
        Takes a path to an image file, returns a dict:
        {"layer2": tensor, "layer3": tensor}
        Each tensor has shape [1, C, H, W].
        """
        image = Image.open(image_path).convert("RGB")
        input_tensor = self.preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # add batch dimension
        input_batch = input_batch.to(self.device)

        self._features = {}  # clear previous results

        with torch.no_grad():
            _ = self.model(input_batch)

        # Return copies so they aren't overwritten by the next call
        return {
            "layer2": self._features["layer2"].clone(),
            "layer3": self._features["layer3"].clone()
        }