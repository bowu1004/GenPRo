import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np
from typing import Tuple, Optional

class ResNet18Encoder(nn.Module):
    """ResNet-18视觉编码器，用于提取图像特征"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 feature_dim: int = 512,
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        """
        初始化ResNet-18编码器
        
        Args:
            input_channels: 输入图像通道数
            feature_dim: 输出特征维度
            pretrained: 是否使用预训练权重
            freeze_backbone: 是否冻结骨干网络
        """
        super(ResNet18Encoder, self).__init__()
        
        self.feature_dim = feature_dim
        
        # 使用torchvision的ResNet-18模型
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
            
        self.backbone = models.resnet18(weights=weights)
        
        # 如果输入通道数不是3，需要修改第一层
        if input_channels != 3:
            self.backbone.conv1 = nn.Conv2d(
                input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # 获取ResNet-18最后一层的输出维度
        backbone_output_dim = self.backbone.fc.in_features
        
        # 替换最后的全连接层
        self.backbone.fc = nn.Identity()
        
        # 添加自定义特征提取头
        self.feature_head = nn.Sequential(
            nn.Linear(backbone_output_dim, feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # 冻结骨干网络参数
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 图像预处理 - 只用于normalize
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像张量 [batch_size, channels, height, width]
            
        Returns:
            features: 提取的特征 [batch_size, feature_dim]
        """
        # 通过骨干网络提取特征
        backbone_features = self.backbone(x)
        
        # 通过特征头生成最终特征
        features = self.feature_head(backbone_features)
        
        return features
        
    def encode_image(self, image: np.ndarray) -> torch.Tensor:
        """
        编码单张图像
        
        Args:
            image: 输入图像 [height, width, channels] (numpy array)
            
        Returns:
            features: 图像特征 [feature_dim]
        """

        # 预处理图像
        if isinstance(image, np.ndarray):
            # 确保图像在0-1范围内
            if image.max() > 1.0:
                image = image.astype(np.float32) / 255.0
            else:
                image = image.astype(np.float32)
                
            # 转换为PyTorch张量 [H, W, C] -> [C, H, W]
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
            
            # 手动resize到224x224
            image_tensor = F.interpolate(image_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            
            # 应用normalize
            image_tensor = self.normalize(image_tensor)
        elif isinstance(image, torch.Tensor):
            # 处理已经是张量的情况
            if image.dim() == 4 and image.shape[1:] == (224, 224, 3):
                # 输入格式为 [B, H, W, C]，需要转换为 [B, C, H, W]
                image_tensor = image.permute(0, 3, 1, 2)
            elif image.dim() == 3 and image.shape == (224, 224, 3):
                # 输入格式为 [H, W, C]，需要转换为 [B, C, H, W]
                image_tensor = image.permute(2, 0, 1).unsqueeze(0)
            else:
                image_tensor = image
            
            # 确保数据类型和范围正确
            if image_tensor.max() > 1.0:
                image_tensor = image_tensor.float() / 255.0
            
            # 应用normalize
            image_tensor = self.normalize(image_tensor)
        else:
            image_tensor = image.unsqueeze(0) if image.dim() == 3 else image
            
        # 确保张量在正确的设备上
        device = next(self.parameters()).device
        image_tensor = image_tensor.to(device)
        
        # 前向传播
        with torch.no_grad():
            features = self.forward(image_tensor)
            
        return features.squeeze(0)  # 移除batch维度
        
    def encode_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码批量图像
        
        Args:
            images: 批量图像 [batch_size, channels, height, width]
            
        Returns:
            features: 批量特征 [batch_size, feature_dim]
        """
        return self.forward(images)


class VisionStateEncoder(nn.Module):
    """视觉-状态融合编码器"""
    
    def __init__(self, 
                 vision_feature_dim: int = 512,
                 state_dim: int = 20,
                 hidden_dim: int = 256,
                 output_dim: int = 256):
        """
        初始化视觉-状态融合编码器
        
        Args:
            vision_feature_dim: 视觉特征维度
            state_dim: 状态向量维度
            hidden_dim: 隐藏层维度
            output_dim: 输出特征维度
        """
        super(VisionStateEncoder, self).__init__()
        
        self.vision_feature_dim = vision_feature_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        
        # 视觉特征处理
        self.vision_processor = nn.Sequential(
            nn.Linear(vision_feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 状态特征处理
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, vision_features: torch.Tensor, 
                state_vector: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            vision_features: 视觉特征 [batch_size, vision_feature_dim]
            state_vector: 状态向量 [batch_size, state_dim]
            
        Returns:
            fused_features: 融合特征 [batch_size, output_dim]
        """
        # 处理视觉特征
        vision_processed = self.vision_processor(vision_features)
        # 确保vision_processed有正确的batch维度
        if vision_processed.dim() == 1:
            vision_processed = vision_processed.unsqueeze(0)
        
        # 处理状态特征
        state_processed = self.state_processor(state_vector)
        
        # 特征融合
        concatenated = torch.cat([vision_processed, state_processed], dim=1)
        fused_features = self.fusion_network(concatenated)
        
        return fused_features


class ImagePreprocessor:
    """图像预处理工具类"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        """
        初始化图像预处理器
        
        Args:
            target_size: 目标图像尺寸 (width, height)
        """
        self.target_size = target_size
        
        # 数据增强变换（训练时使用）
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, 
                                 saturation=0.1, hue=0.05),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 验证变换（推理时使用）
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess(self, image: np.ndarray, training: bool = False) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: 输入图像 [height, width, channels]
            training: 是否为训练模式
            
        Returns:
            processed_image: 预处理后的图像张量
        """
        # 确保图像格式正确
        if isinstance(image, np.ndarray):
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
                
        # 选择合适的变换
        transform = self.train_transform if training else self.val_transform
        
        return transform(image)
        
    def preprocess_batch(self, images: list, training: bool = False) -> torch.Tensor:
        """
        批量预处理图像
        
        Args:
            images: 图像列表
            training: 是否为训练模式
            
        Returns:
            batch_tensor: 批量图像张量 [batch_size, channels, height, width]
        """
        processed_images = []
        for image in images:
            processed = self.preprocess(image, training)
            processed_images.append(processed)
            
        return torch.stack(processed_images)


if __name__ == "__main__":
    # 测试视觉编码器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建编码器
    vision_encoder = ResNet18Encoder(feature_dim=512).to(device)
    
    # 测试单张图像编码
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    features = vision_encoder.encode_image(test_image)
    print(f"单张图像特征形状: {features.shape}")
    
    # 测试批量图像编码
    batch_images = torch.randn(4, 3, 224, 224).to(device)
    batch_features = vision_encoder.encode_batch(batch_images)
    print(f"批量图像特征形状: {batch_features.shape}")
    
    # 测试视觉-状态融合编码器
    fusion_encoder = VisionStateEncoder(
        vision_feature_dim=512,
        state_dim=20,
        output_dim=256
    ).to(device)
    
    # 模拟状态向量
    state_vector = torch.randn(4, 20).to(device)
    fused_features = fusion_encoder(batch_features, state_vector)
    print(f"融合特征形状: {fused_features.shape}")
    
    # 测试图像预处理器
    preprocessor = ImagePreprocessor()
    processed_image = preprocessor.preprocess(test_image, training=True)
    print(f"预处理图像形状: {processed_image.shape}")
    
    print("视觉编码器测试完成！")