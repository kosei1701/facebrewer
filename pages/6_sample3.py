import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from mtcnn import MTCNN
import streamlit as st
from utils.grad_cam import GradCAM  # utilsフォルダに移動したGrad-CAMクラスのインポート

# ディレクトリのベースパスを取得
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# モデルを読み込む関数
def load_model(model_path, num_classes=5):
    model = models.resnet18(pretrained=True)
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

# モデルのパス
model_path = os.path.join(BASE_DIR, 'model', 'resnet_model(5).pth')

# モデルをロード
model_ft = load_model(model_path, num_classes=len(class_names))

# Streamlitアプリケーション
st.title("Grad-CAM可視化による顔分類")
uploaded_file = st.file_uploader("画像を選択してください...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_rgb = np.array(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # MTCNNで顔検出
    detector = MTCNN()
    faces = detector.detect_faces(image_rgb)

    grad_cam = GradCAM(model_ft, model_ft.layer4[1].conv2)

    for face in faces:
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

            for idx, prob in enumerate(probabilities):
                class_name = class_names[idx]
                st.write(f"クラス: {class_name}, 確率: {prob.item():.4f}")

        cam = grad_cam.generate_cam(face_tensor, class_idx=max_idx.item())

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        face_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)

        superimposed_img = cv2.addWeighted(face_bgr, 0.6, heatmap_resized, 0.4, 0)

        box_color = class_colors[class_names[max_idx.item()]]
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), box_color, 2)

        (text_width, text_height), baseline = cv2.getTextSize(class_names[max_idx.item()], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        overlay = image_bgr.copy()
        cv2.rectangle(overlay, (x, y - text_height - baseline), (x + text_width, y), box_color, thickness=cv2.FILLED)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, image_bgr)

        text_x = x
        text_y = y - baseline - (text_height // 2)
        cv2.putText(image_bgr, class_names[max_idx.item()], (text_x, text_y + (text_height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        try:
            image_bgr[y:y+h, x:x+w] = superimposed_img
        except ValueError as e:
            st.write(f"スーパーインポーズ画像の適用エラー: {e}")
            st.write(f"スーパーインポーズ画像の形状: {superimposed_img.shape}, 元の顔ボックスの形状: {(h, w)}")

    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
