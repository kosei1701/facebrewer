# pages/1_Find.py

import streamlit as st
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from mtcnn import MTCNN
import matplotlib.pyplot as plt

from utils.grad_cam import GradCAM
from utils.image_utils import add_transparent_bg
from utils.model_utils import load_model




# サイドバーに "Brew"
st.sidebar.write("Brew")

# クラス名のリストと各クラスに対応する色を定義
class_names = ['Awamori', 'Nihonshu', 'No Alcohol', 'Shochu', 'Too Young', 'Wine']
class_colors = {
    'Awamori': (255, 105, 180),   # 濃いパステルピンク
    'Nihonshu': (144, 238, 144),  # 濃いパステルグリーン
    'No Alcohol': (135, 206, 250), # 濃いパステルブルー
    'Shochu': (255, 218, 185),    # 濃いパステルオレンジ
    'Too Young': (221, 160, 221), # 濃いパステルパープル
    'Wine': (250, 250, 210)       # 濃いパステルイエロー
}

def main():
    st.title('Brew your face')

    # モデルの読み込み
    model_path = os.path.join(BASE_DIR, 'model', 'resnet_model.pth')
    model_ft = load_model(model_path)

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
            # 顔部分の座標を取得
            x, y, w, h = face['box']

            # バウンディングボックスを正方形にする
            size = max(w, h)
            center_x = x + w // 2
            center_y = y + h // 2
            x = max(center_x - size // 2, 0)
            y = max(center_y - size // 2, 0)
            w = h = size

            # 顔部分のみを切り抜く
            face_image = image_rgb[y:y+h, x:x+w]
            face_resized = cv2.resize(face_image, (112, 112))

            # OpenCVからPIL形式に変換
            face_pil = Image.fromarray(face_resized)

            # 画像の変換
            transform = transforms.Compose([
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            face_tensor = transform(face_pil).unsqueeze(0)

            # 推論
            outputs = model_ft(face_tensor.requires_grad_(True))  # requires_gradをTrueに設定
            probabilities = torch.softmax(outputs, dim=1).squeeze()
            max_prob, max_idx = torch.max(probabilities, dim=0)

            # Grad-CAMの生成
            cam = grad_cam.generate_cam(face_tensor, class_idx=max_idx.item())

            # ヒートマップの適用
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (112, 112))
            superimposed_img = heatmap * 0.4 + face_resized

            # 元の顔部分のサイズにリサイズ
            superimposed_img_resized = cv2.resize(superimposed_img, (w, h))

            # 画像にバウンディングボックスとラベルを追加
            box_color = class_colors[class_names[max_idx.item()]]
            cv2.rectangle(image_rgb, (x, y), (x+w, y+h), box_color, 2)
            
            # バウンディングボックス内にクラス名を描画
            add_transparent_bg(image_rgb, class_names[max_idx.item()], (x, y, x+w, y+h))

            # リサイズされた画像を元の画像に適用
            try:
                image_rgb[y:y+h, x:x+w] = superimposed_img_resized
            except ValueError as e:
                print(f"Error applying superimposed image: {e}")
                print(f"Resized shape: {superimposed_img_resized.shape}, original face box shape: {(h, w)}")


        # 画像の上にスペースと「Result」のテキストを追加
        image_rgb = cv2.copyMakeBorder(image_rgb, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(image_rgb, 'Result', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        # 画像表示
        st.image(image_rgb, channels="BGR")

        # グラフの表示
        fig, ax = plt.subplots(figsize=(10, 3))  # 横10:縦3の比率に調整
        colors = [(r / 255, g / 255, b / 255) for r, g, b in [class_colors[class_name] for class_name in class_names]]
        probabilities_np = probabilities.detach().cpu().numpy()
        probabilities_percent = probabilities_np * 100

        # クラス名、確率、色を一緒にしてソート
        sorted_data = sorted(zip(probabilities_percent, class_names, colors), reverse=True)
        sorted_probabilities_percent, sorted_class_names, sorted_colors = zip(*sorted_data)

        bars = ax.barh(sorted_class_names, sorted_probabilities_percent, color=sorted_colors)
        for bar, prob, color in zip(bars, sorted_probabilities_percent, sorted_colors):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{prob:.1f}%', va='center', ha='left', color='black', fontsize=10)

        ax.set_title('Class Probabilities', fontsize=14)
        ax.invert_yaxis()  # グラフを上から大きい順にする
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
