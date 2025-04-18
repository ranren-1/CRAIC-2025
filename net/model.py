import torch
import torch.nn as nn
from torchvision import models

class FundusResNet(nn.Module):
    def __init__(self, num_classes=8):
        super(FundusResNet, self).__init__()
        # 加载预训练的ResNet18模型
        # self.base_model = models.resnet18(pretrained=True)
        self.base_model = models.resnet50(pretrained=True)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, left_images, right_images, labels=None):
        # 处理左眼图像
        outputs_left = self.base_model(left_images)
        # 处理右眼图像
        outputs_right = self.base_model(right_images)

        # 对两只眼睛的预测结果取平均
        outputs = (outputs_left + outputs_right) / 2
        
        if self.training:
            return outputs
        else:
            # 在测试阶段，返回sigmoid后的预测结果
            return torch.sigmoid(outputs)