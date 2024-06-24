import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import streamlit as st
from mtcnn import MTCNN
import matplotlib.pyplot as plt
import os

# ディレクトリのベースパスを取得
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ページ設定
st.set_page_config(
    page_icon=os.path.join(BASE_DIR, 'image', 'favicon1.png')  # ファビコンのパスを設定
)


# サイドバーに "Find" 
st.sidebar.write("Find")


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

def add_transparent_bg(img, text, bbox, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, thickness=2):
    # 文字のサイズを取得
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # バウンディングボックスの範囲内で文字を描画するための左上座標を計算
    text_x = bbox[0]
    text_y = max(bbox[1] - 10, 0)  # バウンディングボックスの上に少しオフセットして配置

    # 文字の背景を半透明の黒で塗りつぶす
    overlay = img.copy()
    cv2.rectangle(overlay, (text_x, text_y + 5), (text_x + text_size[0], text_y - text_size[1] - 5), (0, 0, 0), -1)
    opacity = 0.4  # 透明度を設定
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)

    # バウンディングボックス内に文字を描画
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

def main():
    st.title('Image Processing with Grad-CAM')

    # モデルの読み込みとクラス数の変更
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, 6)  # 6クラスに変更

    # 学習済みの重みをCPUにマップして読み込む
    model_path = os.path.join('model', 'resnet_model.pth')
    model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
