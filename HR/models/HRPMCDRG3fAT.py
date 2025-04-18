import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import yaml
import os
from HR.hrnet.cls_hrnet import get_cls_net, Bottleneck, BN_MOMENTUM

class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x):
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output

class CBAMBlock(nn.Module):
    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ChannelAttention=ChannelAttention(channel=channel,reduction=reduction)
        self.SpatialAttention=SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        B,C,H,W = x.size()
        residual=x
        out=x * self.ChannelAttention(x)
        out=out * self.SpatialAttention(out)
        return out+residual

class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        # 调用父类的构造函数
        super(MSFF, self).__init__()
        # 定义第一个卷积序列
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),  # 1x1卷积
            nn.BatchNorm2d(inchannel),  # 批归一化
            nn.ReLU(inplace=True)  # ReLU激活
        )
        # 定义第二个卷积序列，使用3x3卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),  # 1x1卷积降维
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1, bias=False),  # 3x3卷积
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),  # 1x1卷积升维
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 定义第三个卷积序列，使用5x5卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),  # 5x5卷积
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 定义第四个卷积序列，使用7x7卷积
        self.conv4 = nn.Sequential(
            nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 7, stride=1, padding=3, bias=False),  # 7x7卷积
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        # 定义混合卷积序列
        self.convmix = nn.Sequential(
            nn.Conv2d(4 * inchannel, inchannel, 1, stride=1, bias=False),  # 1x1卷积降维
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False),  # 3x3卷积
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 通过不同的卷积序列
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        # 在通道维度上拼接
        x_f = torch.cat([x1, x2, x3, x4], dim=1)
        # 通过混合卷积序列
        out = self.convmix(x_f)

        # 返回输出
        return out

class PagFM(nn.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=True, BatchNorm=nn.BatchNorm2d):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu

        # self.f_x = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
        #     BatchNorm(mid_channels)
        # )
        # self.f_y = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
        #     BatchNorm(mid_channels)
        # )

        self.f_x = MSFF(in_channels, mid_channels)
        self.f_y = MSFF(in_channels, mid_channels)

        if with_channel:
            self.up = nn.Sequential(
                nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
                BatchNorm(in_channels)
            )
        if after_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        input_size = x.size()
        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        x_k = self.f_x(x)
        
        if self.with_channel:
            sim_map = torch.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
        y = F.interpolate(y, size=[input_size[2], input_size[3]], mode='bilinear', align_corners=False)
        x = (1 - sim_map) * x + sim_map * y
        return x

# 图卷积层类,继承自nn.Module
class GraphConvolution(nn.Module):
    # 初始化函数
    def __init__(self, in_dim, out_dim):
        # 调用父类初始化
        super(GraphConvolution, self).__init__()
        # 定义LeakyReLU激活函数,负斜率为0.2
        self.relu = nn.LeakyReLU(0.2)
        # 定义1D卷积层,输入维度为in_dim,输出维度为out_dim,卷积核大小为1
        self.weight = nn.Conv1d(in_dim, out_dim, 1)

    # 前向传播函数
    def forward(self, adj, nodes):
        # 将节点特征与邻接矩阵相乘
        nodes = torch.matmul(nodes, adj)
        # 经过LeakyReLU激活
        nodes = self.relu(nodes)
        # 经过卷积层变换
        nodes = self.weight(nodes)
        # 再次经过LeakyReLU激活
        nodes = self.relu(nodes)
        # 返回处理后的节点特征
        return nodes

class HRPMCDRG3fAT(nn.Module):
    def __init__(self, num_classes=8):
        num_classes = 8
        # 继承nn.Module类并初始化
        super(HRPMCDRG3fAT, self).__init__()
        # 构建配置文件路径
        config_path = 'HR/hrnet/cls_hrnet_w18_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
        # 打开并读取配置文件
        with open(config_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        
        # 加载HRNet模型
        self.base_model = get_cls_net(cfg)
        # 获取stage4的输出通道数
        _, pre_stage_channels = self.base_model._make_stage(
            self.base_model.stage4_cfg,
            [32, 64, 128, 256],
            True
        )

        # 构建预训练权重文件路径
        pretrained = 'HR/hrnet/HRNet_W18_C_pretrained.pth'
        try:
            # 加载预训练权重
            state_dict = torch.load(pretrained)
            # 获取当前模型的状态字典
            model_dict = self.base_model.state_dict()
            # 筛选出形状匹配的权重
            matched_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            # 找出不匹配的权重
            unmatched = set(state_dict.keys()) - set(matched_dict.keys())
            if unmatched:
                print(f"以下权重因结构不匹配而跳过: {unmatched}")
            # 更新模型权重
            model_dict.update(matched_dict)
            # 加载筛选后的权重
            self.base_model.load_state_dict(model_dict, strict=False)
            print(f"成功加载匹配的预训练权重")
        except Exception as e:
            # 处理加载失败的情况
            print(f"加载预训练权重时出错: {e}")
            print("将使用随机初始化的权重继续训练")

        # 初始化PagFM模块列表，用于特征融合
        self.pagfm_modules = nn.ModuleList([
            PagFM(in_channels=18, mid_channels=18),
            PagFM(in_channels=36, mid_channels=36),
            PagFM(in_channels=72, mid_channels=72),
            PagFM(in_channels=144, mid_channels=144)
        ])

        # 创建头部网络结构
        self.incre_modules, self.downsamp_modules, \
            self.final_layer = self.base_model._make_head(pre_stage_channels)
        
        self.cbam = CBAMBlock(channel=2048, reduction=16, kernel_size=7)

        # 创建全局平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 创建最终的全连接层，输入2048通道，输出为类别数
        self.fc = nn.Linear(2048, num_classes)
    
        self.gcn_dim = 512                
        self.in_planes = 2048         

        self.constraint_classifier = nn.Conv2d(self.in_planes, num_classes, (1, 1), bias=False)

        self.gcn_dim_transform = nn.Conv2d(self.in_planes, self.gcn_dim, (1, 1))

        self.transformer_dim = 512

        self.num_classes = num_classes

        # 矩阵变换层
        self.matrix_transform = nn.Conv1d(self.gcn_dim, self.num_classes, 1)

        # 图卷积层
        self.forward_gcn = GraphConvolution(self.gcn_dim, self.gcn_dim)

        # 掩码矩阵和GCN分类器
        self.mask_mat = nn.Parameter(torch.eye(self.num_classes).float())
        self.gcn_classifier = nn.Conv1d(self.gcn_dim, self.num_classes, 1)

        self.cbam = CBAMBlock(channel=2048, reduction=16, kernel_size=7)
   
    def build_nodes(self, x):
        # 通过约束分类器处理输入x,生成掩码
        mask = self.constraint_classifier(x)
        # 重塑掩码维度为(batch_size, num_classes, -1)
        mask = mask.view(mask.size(0), mask.size(1), -1)
        # 对掩码进行sigmoid激活,将值压缩到0-1之间
        mask = torch.sigmoid(mask)
        # 转置掩码维度,将类别维度和特征维度交换
        mask = mask.transpose(1, 2)

        # 通过GCN维度转换层处理输入x
        x = self.gcn_dim_transform(x)
        # 重塑x的维度为(batch_size, gcn_dim, -1)
        x = x.view(x.size(0), x.size(1), -1)
        # 将转换后的x与掩码相乘,得到图特征v_g
        nodes = torch.matmul(x, mask)

        # 返回最终的节点特征
        return nodes

    def build_joint_correlation_matrix(self, x):
        # 通过矩阵变换层生成关联矩阵
        joint_correlation = self.matrix_transform(x)
        # 对关联矩阵进行sigmoid激活,将值压缩到0-1之间
        joint_correlation = torch.sigmoid(joint_correlation)
        # 返回最终的关联矩阵
        return joint_correlation

    def forward(self, left_images, right_images, labels=None):            
        # 提取左眼特征图
        left_list = self.base_model(left_images)
        # 提取右眼特征图
        right_list = self.base_model(right_images)

        # 应用第一个增量模块
        y = self.incre_modules[0](self.pagfm_modules[0](left_list[0], right_list[0]))
        # 遍历所有下采样模块
        # print(y.shape)
        for i in range(len(self.downsamp_modules)):
            # 融合特征：增量模块输出 + 下采样模块输出
            # print(y.shape)
            y = self.incre_modules[i+1](self.pagfm_modules[i+1](left_list[i+1], right_list[i+1])) + \
                        self.downsamp_modules[i](y)

        y = self.final_layer(y)
        # print(y.shape)

        # y = self.cbam(y)

        V = self.build_nodes(y)
        # print(V.shape)

        A_s = self.build_joint_correlation_matrix(V)
        # 通过图卷积网络更新节点特征
        G = self.forward_gcn(A_s, V) + V
        # 使用图分类器得到输出
        out_gcn = self.gcn_classifier(G)
        # 获取掩码矩阵并阻止梯度回传
        mask_mat = self.mask_mat.detach()
        # 将输出与掩码矩阵相乘并在最后一个维度求和
        out_gcn = (out_gcn * mask_mat).sum(-1)

        # 全局平均池化
        pooled_features = self.avgpool(y)
        pooled_features = torch.flatten(pooled_features, 1)
        
        # 通过全连接层得到分类结果
        outputs = self.fc(pooled_features)

        weight = 0.3
        if self.training:
            return (1-weight) * outputs + weight * out_gcn
        else:
            return torch.sigmoid((1-weight) * outputs + weight * out_gcn)
