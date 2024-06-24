import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import streamlit as st
from mtcnn import MTCNN


# クラス名のリストと各クラスに対応する色を定義
class_names = ['Awamori', 'Nihonshu', 'No Alcohol', 'Shochu', 'Too Young', 'Wine']
class_colors = {
    'Awamori': (255, 0, 0),    # 赤
    'Nihonshu': (0, 255, 0),   # 緑
    'No Alcohol': (0, 0, 255), # 青
    'Shochu': (255, 255, 0),   # 黄色
    'Too Young': (255, 0, 255), # ピンク
    'Wine': (0, 255, 255)      # シアン
}

# Grad-CAMの定義
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, class_idx=None):
        if class_idx is None:
            class_idx = input_tensor.argmax(dim=1).item()

        self.model.zero_grad()
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward()

        gradients = self.gradients
        activations = self.activations

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)
        heatmap = cv2.resize(heatmap, (input_tensor.shape[2], input_tensor.shape[3]))
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        return heatmap

def main():
    st.title('Image Processing with Grad-CAM')

    # モデルの読み込みとクラス数の変更
    model_ft = models.resnet18(weights=None)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, 6)  # 6クラスに変更

    # 学習済みの重みをCPUにマップして読み込む
    model_ft.load_state_dict(torch.load(r'C:\Users\kosei\Desktop\成果物\成果物\env\resnet_model.pth', map_location=torch.device('cpu')))
    model_ft.eval()

    # Grad-CAMのインスタンス化
    target_layer = model_ft.layer4[1].conv2
    grad_cam = GradCAM(model_ft, target_layer)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        # 画像をOpenCVで読み込む
        image = np.array(Image.open(uploaded_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # MTCNNで顔検出
        detector = MTCNN()
        faces = detector.detect_faces(image_rgb)

        # 顔が検出された場合は推論を行う
        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            size = max(w, h)
            center_x = x + w // 2
            center_y = y + h // 2
            x = max(center_x - size // 2, 0)
            y = max(center_y - size // 2, 0)
            w = h = size

            face_image = image_rgb[y:y+h, x:x+w]
            face_resized = cv2.resize(face_image, (112, 112))
            face_pil = Image.fromarray(face_resized)

            transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            face_tensor = transform(face_pil).unsqueeze(0)

            with torch.no_grad():
                outputs = model_ft(face_tensor)
                probabilities = torch.softmax(outputs, dim=1).squeeze()
                max_prob, max_idx = torch.max(probabilities, dim=0)

                # 結果の表示
                for idx, prob in enumerate(probabilities):
                    class_name = class_names[idx]
                    st.write(f"Class: {class_name}, Probability: {prob.item():.4f}")

            cam = grad_cam.generate_cam(face_tensor, class_idx=max_idx.item())

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (112, 112))
            superimposed_img = heatmap * 0.4 + face_resized

            superimposed_img_resized = cv2.resize(superimposed_img, (w, h))

            box_color = class_colors[class_names[max_idx.item()]]
            cv2.rectangle(image_rgb, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(image_rgb, class_names[max_idx.item()], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

            try:
                image_rgb[y:y+h, x:x+w] = superimposed_img_resized
            except ValueError as e:
                print(f"Error applying superimposed image: {e}")
                print(f"Resized shape: {superimposed_img_resized.shape}, original face box shape: {(h, w)}")

        # 結果の画像を表示
        st.image(image_rgb, channels="BGR")

if __name__ == "__main__":
    main()
