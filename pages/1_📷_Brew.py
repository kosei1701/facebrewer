import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from mtcnn import MTCNN
import streamlit as st
import matplotlib.pyplot as plt
import sys
import io

# Grad-CAMのクラス定義
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

# ディレクトリのベースパスを取得
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ページ設定
st.set_page_config(
    page_icon=os.path.join(BASE_DIR, 'image', 'favicon3.png')  # ファビコンのパスを設定
)

# サイドバーに "Brew"
st.sidebar.write("Brew")

# モデルを読み込む関数
def load_model(model_path, num_classes=5):
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# クラス名と色の定義
class_names = ['Awamori', 'Nihonshu', 'No Alcohol', 'Shochu', 'Wine']
class_colors = {
    'Awamori': (147, 86, 219),    # 落ち着いたピンク (BGR)
    'Nihonshu': (96, 183, 99),    # 落ち着いたグリーン (BGR)
    'No Alcohol': (230, 153, 60), # 落ち着いたブルー (BGR)
    'Shochu': (112, 176, 255),    # 落ち着いたオレンジ (BGR)
    'Wine': (166, 232, 232)       # 落ち着いたイエロー (BGR)
}

def bgr_to_rgba(color, alpha=1.0):
    r, g, b = color
    return (r / 255, g / 255, b / 255, alpha)

# モデルのパス
model_path = os.path.join(BASE_DIR, 'model', 'resnet_model(5).pth')

# モデルをロード
model_ft = load_model(model_path, num_classes=len(class_names))

# Streamlitアプリケーション
st.title("Image brew")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.divider()
    st.write("#### Detection")

    image = Image.open(uploaded_file)
    image_rgb = np.array(image)

    # RGBA画像の場合はRGBに変換
    if image_rgb.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # MTCNNで顔検出
    detector = MTCNN()

    # stdoutとstderrのリダイレクト
    sys_stdout = sys.stdout
    sys_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        faces = detector.detect_faces(image_rgb)
    finally:
        sys.stdout = sys_stdout
        sys.stderr = sys_stderr

    grad_cam = GradCAM(model_ft, model_ft.layer4[1].conv2)

    if len(faces) == 0:
        st.write("No face detected")
    else:
        for i, face in enumerate(faces):
            x, y, w, h = face['box']

            size = max(w, h)
            center_x = x + w // 2
            center_y = y + h // 2
            x = max(center_x - size // 2, 0)
            y = max(center_y - size // 2, 0)
            w = h = size

            face_image = image_rgb[y:y+h, x:x+w]  # RGB画像をそのまま使う

            face_pil = Image.fromarray(face_image)

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

            # Grad-CAMの生成
            cam = grad_cam.generate_cam(face_tensor, class_idx=max_idx.item())
            cam = cv2.resize(cam, (w, h))

            # ヒートマップをRGBに変換して色を逆転
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # BGRからRGBに変換
            heatmap_inverted = 255 - heatmap  # 色の逆転

            # 元の画像に重ね合わせる
            superimposed_img = cv2.addWeighted(face_image, 0.6, heatmap_inverted, 0.4, 0)

            # バウンディングボックスの描画
            box_color = class_colors[class_names[max_idx.item()]]
            cv2.rectangle(image_bgr, (x, y), (x+w, y+h), box_color, 2)

            # クラスラベルの四角とテキストを描画
            (text_width, text_height), baseline = cv2.getTextSize(class_names[max_idx.item()], cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)
            overlay = image_bgr.copy()
            cv2.rectangle(overlay, (x, y - text_height - baseline), (x + text_width, y), box_color, thickness=cv2.FILLED)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, image_bgr)
            text_x = x
            text_y = y - baseline - (text_height // 2)
            cv2.putText(image_bgr, class_names[max_idx.item()], (text_x, text_y + (text_height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=4)

            try:
                image_bgr[y:y+h, x:x+w] = superimposed_img
            except ValueError as e:
                st.write(f"スーパーインポーズ画像の適用エラー: {e}")
                st.write(f"スーパーインポーズ画像の形状: {superimposed_img.shape}, 元の顔ボックスの形状: {(h, w)}")

            # 画像とグラフを並べて表示
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(face_image, caption=f"Detected Face {i+1}", use_column_width=True)
            with col2:
                fig, ax = plt.subplots()
                fig.set_size_inches(10, 3)  # 横と縦のサイズを設定
                ax.barh(class_names, probabilities, color=[bgr_to_rgba(class_colors[name]) for name in class_names])
                ax.set_xlim([0, 1])
                ax.set_xlabel('Probability')

                # 各バーの横に確率をパーセンテージ表示
                for j in range(len(probabilities)):
                    ax.text(probabilities[j] + 0.01, j, f"{probabilities[j]*100:.1f}%", va='center')

                st.pyplot(fig)

        st.divider()
        st.write("#### Result")

        # 加工した元画像の表示
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

        # 画像のダウンロードボタンを追加
        result_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_img)
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG")
        buffer.seek(0)
        st.download_button(
        label="Download",
        data=buffer,
        file_name="processed_image.png",
        mime="image/png"
       )
