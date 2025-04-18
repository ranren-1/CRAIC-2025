import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from net.model import FundusResNet
from PIL import Image
from tqdm import tqdm

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # 使用 register_full_backward_hook 替代 register_backward_hook
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensors, class_idx):
        left_tensor, right_tensor = input_tensors
        self.model.zero_grad()
        output = self.model(left_tensor, right_tensor)
        target = output[:, class_idx]
        target.backward()

        gradients = self.gradients.cpu().data.numpy()
        activations = self.activations.cpu().data.numpy()

        weights = np.mean(gradients, axis=(2, 3))
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (left_tensor.shape[2], left_tensor.shape[3]))

        # 修复归一化问题，避免除以零
        if np.max(cam) > 0:
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
        else:
            cam = np.zeros_like(cam)

        return cam

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = transform(image).unsqueeze(0)
    return image

def overlay_cam_on_image(image_path, cam, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (cam.shape[1], cam.shape[0]))
    cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    overlay = 0.6 * image + 0.4 * cam
    cv2.imwrite(output_path, np.uint8(overlay))

def generate_grad_cam(left_image_path, right_image_path, model_path, output_dir, class_idx=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FundusResNet(num_classes=8)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval().to(device)

    target_layer = model.base_model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    left_tensor = preprocess_image(left_image_path).to(device)
    right_tensor = preprocess_image(right_image_path).to(device)

    left_cam = grad_cam.generate_cam((left_tensor, right_tensor), class_idx)
    right_cam = grad_cam.generate_cam((right_tensor, left_tensor), class_idx)

    left_output_path = os.path.join(output_dir, "left_eye_cam.jpg")
    right_output_path = os.path.join(output_dir, "right_eye_cam.jpg")
    overlay_cam_on_image(left_image_path, left_cam, left_output_path)
    overlay_cam_on_image(right_image_path, right_cam, right_output_path)

    print(f"Grad-CAM visualizations saved to {left_output_path} and {right_output_path}")

def generate_grad_cam_for_directory(img_dir, model_path, output_dir, class_idx=0):
    """
    对 img_dir 中的所有图片生成 Grad-CAM 可视化。

    Args:
        img_dir (str): 包含左右眼图片的文件夹路径。
        model_path (str): 模型权重文件路径。
        output_dir (str): 输出 Grad-CAM 可视化图片的文件夹路径。
        class_idx (int): 目标类别索引。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FundusResNet(num_classes=8)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval().to(device)

    target_layer = model.base_model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    os.makedirs(output_dir, exist_ok=True)
    image_files = sorted(os.listdir(img_dir))

    if len(image_files) % 2 != 0:
        raise ValueError("图片数量不是偶数，请确保左右眼图片成对出现。")

    for i in tqdm(range(0, len(image_files), 2), desc="Generating Grad-CAM"):
        try:
            left_image_path = os.path.join(img_dir, image_files[i])
            right_image_path = os.path.join(img_dir, image_files[i + 1])

            left_tensor = preprocess_image(left_image_path).to(device)
            right_tensor = preprocess_image(right_image_path).to(device)

            left_cam = grad_cam.generate_cam((left_tensor, right_tensor), class_idx)
            right_cam = grad_cam.generate_cam((right_tensor, left_tensor), class_idx)

            left_output_path = os.path.join(output_dir, f"{os.path.splitext(image_files[i])[0]}_cam.jpg")
            right_output_path = os.path.join(output_dir, f"{os.path.splitext(image_files[i + 1])[0]}_cam.jpg")

            overlay_cam_on_image(left_image_path, left_cam, left_output_path)
            overlay_cam_on_image(right_image_path, right_cam, right_output_path)

        except Exception as e:
            print(f"Error processing {image_files[i]} and {image_files[i + 1]}: {e}")

    print(f"Grad-CAM visualizations saved to {output_dir}")

if __name__ == "__main__":
    img_dir = "Data/testimag/img"
    model_path = "net/checkpoints/net50-new_20250416_105343/best_model.pth"
    output_dir = "Data/testimag/grad_cam_outputs"
    generate_grad_cam_for_directory(img_dir, model_path, output_dir, class_idx=0)
