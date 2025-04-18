import torch
from torchvision import transforms
from HR.models.HRPMCDRG3fAT import HRPMCDRG3fAT
import os
from PIL import Image
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from net.grad_cam import GradCAM,generate_grad_cam
from net.grad_cam import generate_grad_cam_for_directory

def load_model(device, model_path="HR/models/best_model.pth"):
    """
    加载模型并加载权重。
    """
    model = HRPMCDRG3fAT()
    try:
        # 显式设置 weights_only=True
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        raise FileNotFoundError(f"模型文件未找到：{model_path}")
    except RuntimeError as e:
        raise RuntimeError(f"加载模型权重失败，请检查模型结构是否匹配。错误信息：{e}")
    model.to(device)
    model.eval()
    return model


def batch_inference(image_dir, model, transform, device='cpu', threshold=0.5):
    """
    批量处理图片并返回分类结果。
    """
    results = []
    image_files = sorted(os.listdir(image_dir))

    if len(image_files) % 2 != 0:
        raise ValueError("图片数量不是偶数，请确保左右眼图片成对出现。")

    with torch.no_grad():
        for i in tqdm(range(0, len(image_files), 2), desc="Processing images"):
            try:
                left_image_path = os.path.join(image_dir, image_files[i])
                right_image_path = os.path.join(image_dir, image_files[i + 1])

                # 检查文件是否存在
                if not os.path.exists(left_image_path) or not os.path.exists(right_image_path):
                    raise FileNotFoundError(f"文件未找到：{left_image_path} 或 {right_image_path}")

                # 加载图像并转换为 RGB 格式
                left_image = cv2.imread(left_image_path)
                right_image = cv2.imread(right_image_path)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

                # 转换为 PIL 图像并应用 transform
                left_image = transform(Image.fromarray(left_image)).unsqueeze(0).to(device)
                right_image = transform(Image.fromarray(right_image)).unsqueeze(0).to(device)

                # 模型推理
                outputs = model(left_image, right_image).cpu().numpy()

                # 确保概率以百分制显示并保留一位小数
                probabilities = np.round(outputs * 100, 1).tolist()
                print(f"Probabilities for {image_files[i]} and {image_files[i + 1]}: {probabilities}")

                # 预测标签
                predicted_labels = (outputs > threshold).astype(int).tolist()

                results.append((image_files[i], image_files[i + 1], predicted_labels))
            except Exception as e:
                print(f"Error processing {image_files[i]} and {image_files[i + 1]}: {e}")
    
    return results


def calculate_accuracy(results, annotation_file):
    """
    计算推理结果的准确率。
    """
    # 加载注释文件并检查列名
    annotations = pd.read_csv(annotation_file)
    print(f"注释文件的列名: {annotations.columns.tolist()}")  # 打印列名以便调试
    if 'Left-Fundus' not in annotations.columns or 'Right-Fundus' not in annotations.columns:
        raise KeyError(f"注释文件中缺少 'Left-Fundus' 或 'Right-Fundus' 列，请检查文件格式。当前列名: {annotations.columns.tolist()}")

    total = 0
    correct = 0

    for left_image_name, right_image_name, predicted_labels in results:
        try:
            # 获取左右眼的真实标签
            left_annotation = annotations[annotations['Left-Fundus'] == left_image_name].iloc[0, 2:].values.astype(int)
            right_annotation = annotations[annotations['Right-Fundus'] == right_image_name].iloc[0, 2:].values.astype(int)

            # 确保预测标签和真实标签的类型一致，并将预测标签转换为一维数组
            predicted_labels = np.array(predicted_labels, dtype=int).flatten()

            # 打印真实标签和预测标签以便调试
            print(f"Left Image: {left_image_name}, Predicted: {predicted_labels}, Actual (Left): {left_annotation}")
            print(f"Right Image: {right_image_name}, Predicted: {predicted_labels}, Actual (Right): {right_annotation}")

            # 检查预测结果是否与左右眼的真实标签一致
            if np.array_equal(predicted_labels, left_annotation) and np.array_equal(predicted_labels, right_annotation):
                correct += len(predicted_labels)

            total += len(predicted_labels)
        except IndexError:
            print(f"未找到图片 {left_image_name} 或 {right_image_name} 的注释信息，跳过。")

    return correct / total if total > 0 else 0


if __name__ == "__main__":
    # 文件路径相关代码集中到一起

    #输入图片链接
    image_dir = "Data/testimag/img"


    model_path = "net/checkpoints/net50-new_20250416_105343/best_model.pth"
    output_dir = "grad_cam_outputs"

    # 加载设备和模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        results = batch_inference(image_dir, model, transform, device, threshold=0.50)
    except ValueError as e:
        print(f"推理失败：{e}")
        results = []

    # 调用 Grad-CAM 可视化生成
    generate_grad_cam_for_directory(image_dir, model_path, output_dir, class_idx=0)
