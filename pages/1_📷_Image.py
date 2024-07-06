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
from utils.model_utils import load_model

# ディレクトリのベースパスを取得
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ページ設定
st.set_page_config(
    page_icon=os.path.join(BASE_DIR, 'image', 'favicon3.png')  # ファビコンのパスを設定
)

# サイドバーに "Brew"
st.sidebar.write("Brew")

# クラス名のリストと各クラスに対応する色を定義 (BGR形式)
class_names = ['Awamori', 'Nihonshu', 'No Alcohol', 'Shochu', 'Wine']
class_colors = {
    'Awamori': (147, 86, 219),    # 落ち着いたピンク (BGR)
    'Nihonshu': (96, 183, 99),    # 落ち着いたグリーン (BGR)
    'No Alcohol': (230, 153, 60), # 落ち着いたブルー (BGR)
    'Shochu': (112, 176, 255),    # 落ち着いたオレンジ (BGR)
    'Wine': (166, 232, 232)       # 落ち着いたイエロー (BGR)
}

def main():
    st.title('Image brew')

    # モデルの読み込み
    model_path = os.path.join(BASE_DIR, 'model', 'resnet_model(5).pth')
    model_ft = load_model(model_path, num_classes=5)

    # Grad-CAMのインスタンス化
    target_layer = model_ft.layer4[1].conv2
    grad_cam = GradCAM(model_ft, target_layer)

    uploaded_file = st.file_uploader("", type=["jpg", "png"])

    if uploaded_file is not None:
        # 画像をOpenCVで読み込む
        image = np.array(Image.open(uploaded_file))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # MTCNNで顔検出
        detector = MTCNN()
        faces = detector.detect_faces(image_rgb)

        if not faces:
            st.warning("No face detected.")
        else:
            st.write("### Result")

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

                # クラスインデックスを取得
                class_idx = torch.argmax(probabilities).item()
                class_name = class_names[class_idx]

                # Grad-CAMの生成
                cam = grad_cam.generate_cam(face_tensor, class_idx)

                # ヒートマップの適用
                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.resize(heatmap, (w, h))

                # ヒートマップを元の画像に重ねる
                alpha = 0.5  # 透明度
                overlay = image_rgb[y:y+h, x:x+w] * (1 - alpha) + heatmap * alpha
                image_rgb[y:y+h, x:x+w] = overlay.astype(np.uint8)

                # 画像にバウンディングボックスとラベルを追加
                box_color = class_colors[class_name]
                cv2.rectangle(image_rgb, (x, y), (x+w, y+h), box_color, 2)

                # バウンディングボックス内にクラス名を描画
                font_scale = w / 150  # 画像の幅に基づいて文字の大きさを調整
                font_thickness = max(1, w // 300)  # 画像の幅に基づいて文字の細さを調整
                label_bg_color = class_colors[class_name]  # 背景色をクラスの色に設定
                label_bg_alpha = 0.6  # 背景の透明度を設定 (0:透明, 1:不透明)

                # クラス名の背景を描画
                label_size, _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=font_thickness)
                label_w = label_size[0]
                label_h = label_size[1]
                label_bg_start = (x, y - label_h - 10)
                label_bg_end = (x + label_w, y)
                overlay = image_rgb.copy()
                cv2.rectangle(overlay, label_bg_start, label_bg_end, label_bg_color, -1)
                image_rgb = cv2.addWeighted(overlay, label_bg_alpha, image_rgb, 1 - label_bg_alpha, 0)

                # クラス名を描画
                cv2.putText(image_rgb, class_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness + 1, cv2.LINE_AA)

                # 画像とグラフを並べて表示
                cols = st.columns(2)
                with cols[0]:
                    st.image(face_resized, channels="BGR", caption=f"Face {i+1}")

                with cols[1]:
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

            # 最終的な結果の画像を表示
            st.image(image_rgb, channels="BGR")

if __name__ == "__main__":
    main()
