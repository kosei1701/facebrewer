import sys
import os
import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
from mtcnn import MTCNN
import streamlit as st
import matplotlib.pyplot as plt

# BASE_DIRを設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.grad_cam import GradCAM  # utilsフォルダに移動したGrad-CAMクラスのインポート

# ページ設定
st.set_page_config(
    page_icon=os.path.join(BASE_DIR, 'image', 'favicon3.png')  # ファビコンのパスを設定
)

# サイドバーに "Brew"
st.sidebar.write("Brew")

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
st.title("Image brew")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_rgb = np.array(image)

    # RGBA画像の場合はRGBに変換
    if image_rgb.shape[2] == 4:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # MTCNNで顔検出
    detector = MTCNN()
    faces = detector.detect_faces(image_rgb)

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

            cam = grad_cam.generate_cam(face_tensor, class_idx=max_idx.item())

            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

            superimposed_img = cv2.addWeighted(face_image, 0.6, heatmap_resized, 0.4, 0)

            box_color = class_colors[class_names[max_idx.item()]]
            cv2.rectangle(image_bgr, (x, y), (x+w, y+h), box_color, 2)

            # クラスラベルの文字列のサイズを計算
            (text_width, text_height), baseline = cv2.getTextSize(class_names[max_idx.item()], cv2.FONT_HERSHEY_SIMPLEX, 2.0, 4)  # 2倍に
            overlay = image_bgr.copy()

            # 四角のサイズを文字のサイズに合わせて調整
            cv2.rectangle(overlay, (x, y - text_height - baseline), (x + text_width, y), box_color, thickness=cv2.FILLED)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0, image_bgr)

            text_x = x
            text_y = y - baseline - (text_height // 2)

            # テキスト描画時にthicknessを設定する必要があります
            cv2.putText(image_bgr, class_names[max_idx.item()], (text_x, text_y + (text_height // 2)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), thickness=4)

            try:
                image_bgr[y:y+h, x:x+w] = superimposed_img
            except ValueError as e:
                st.write(f"スーパーインポーズ画像の適用エラー: {e}")
                st.write(f"スーパーインポーズ画像の形状: {superimposed_img.shape}, 元の顔ボックスの形状: {(h, w)}")


            # 画像とグラフを並べて表示
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(face_image, channels="RGB", caption=f"Face {i+1}")

            with col2:
                fig, ax = plt.subplots(figsize=(11, 3))  # 横11:縦3の比率に調整
                colors = [(b / 255, g / 255, r / 255) for r, g, b in [class_colors[class_name] for class_name in class_names]]
                probabilities_np = probabilities.detach().cpu().numpy()
                probabilities_percent = probabilities_np * 100

                # クラス名、確率、色を一緒にしてソート
                sorted_data = sorted(zip(probabilities_percent, class_names, colors), reverse=True)
                sorted_probabilities_percent, sorted_class_names, sorted_colors = zip(*sorted_data)

                bars = ax.barh(sorted_class_names, sorted_probabilities_percent, color=sorted_colors)
                for bar, prob, color in zip(bars, sorted_probabilities_percent, sorted_colors):
                    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{prob:.1f}%', va='center', ha='left', color='black', fontsize=10)

                ax.invert_yaxis()  # グラフを上から大きい順にする
                ax.tick_params(axis='both', which='major', labelsize=12)
                plt.tight_layout()
                st.pyplot(fig)

        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)
