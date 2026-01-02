import timm
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class Model(torch.nn.Module):
    def __init__(self, model_name='convnextv2_large.fcmae_ft_in22k_in1k', num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)

        self.data_config = timm.data.resolve_model_data_config(self.backbone)

        self.backbone.reset_classifier(num_classes=num_classes)
    
    def get_transforms(self, is_training=False):
        # return timm.data.create_transform(**self.data_config, is_training=is_training)
        return transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=False),
        ])
        
    def freeze_layers(self, num_frozen_blocks=0):
        """
        Freeze the first `num_frozen_blocks` transformer blocks
        and keep the rest (including head) trainable.
        """
        # Freeze patch embedding
        for param in self.backbone.stem.parameters():
            param.requires_grad = False

        # Freeze the first N transformer blocks
        for i, block in enumerate(self.backbone.stages):
            if i < num_frozen_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            else:
                for param in block.parameters():
                    param.requires_grad = True

        for param in self.backbone.head.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = Model("convnextv2_large.fcmae_ft_in22k_in1k", num_classes=1)
    print(model)