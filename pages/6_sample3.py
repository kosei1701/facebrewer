import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from utils.grad_cam import GradCAM  # Grad-CAMクラスのインポート
import os

# モデルのパス
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'resnet_model(5).pth')

# クラス名と色の定義
class_names = ['Awamori', 'Nihonshu', 'No Alcohol', 'Shochu', 'Wine']
class_colors = {'Awamori': (139, 0, 139), 'Nihonshu': (255, 0, 255), 'No Alcohol': (0, 0, 0), 'Shochu': (75, 0, 130), 'Wine': (128, 0, 0)}

# モデルのロード
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model

# Grad-CAMを生成する関数
def generate_grad_cam(model, image, target_layer):
    grad_cam = GradCAM(model=model)
    mask, heatmap = grad_cam(image, target_layer)
    return mask, heatmap

# 画像を前処理する関数
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Streamlitアプリケーションの定義
def main():
    st.title('Face Classification and Grad-CAM Visualization')
    st.sidebar.title('Settings')
    file_uploaded = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if file_uploaded is not None:
        # 画像を読み込み、表示
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # 顔検出器を初期化
        detector = MTCNN()

        # 画像をRGBに変換
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 顔を検出し、各顔に対して処理
        faces = detector.detect_faces(image_rgb)
        for face in faces:
            # 顔部分を切り取り
            x, y, w, h = face['box']
            face_image = image_rgb[y:y+h, x:x+w]

            # 画像を前処理
            transformed_image = preprocess_image(Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)))

            # モデルをロードして推論
            model = load_model(model_path)
            with torch.no_grad():
                outputs = model(transformed_image)
                probs = F.softmax(outputs, dim=1)
                top_probs, top_classes = probs.topk(1, dim=1)

            # Grad-CAMを生成
            target_layer = model.layer4[-1]  # 例としてResNetの最終レイヤーを指定
            mask, heatmap = generate_grad_cam(model, transformed_image, target_layer)

            # ヒートマップを元の画像に重ねる
            heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
            heatmap_rgba = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_rgba, cv2.COLOR_RGBA2RGB)  # RGBAからRGBに変換

            blended = cv2.addWeighted(heatmap_rgb, 0.5, face_image, 0.5, 0, dtype=cv2.CV_8U)
            blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

            # 画像をStreamlitで表示する
            st.image(blended_bgr, caption='Grad-CAM Heatmap Overlay', use_column_width=True)

            # 分類結果を表示
            st.write(f"Predicted Class: {class_names[top_classes.item()]}")
            st.write(f"Probability: {top_probs.item():.4f}")

if __name__ == '__main__':
    main()
