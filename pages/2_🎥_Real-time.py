import streamlit as st
import cv2
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from utils.model_utils import load_model  # モデルを読み込むための関数を想定
import os

# ディレクトリのベースパスを取得
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ページ設定
st.set_page_config(
    page_icon=os.path.join(BASE_DIR, 'image', 'favicon3.png')  # ファビコンのパスを設定
)

# サイドバーに "Blueprint"
st.sidebar.write("Blueprint")

# OpenCVの顔検出器の準備
face_cascade_path = 'model/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# ResNetモデルの読み込み
model_path = 'model/resnet_model(5).pth'
model = load_model(model_path, num_classes=5)  # モデルを読み込む関数を使用すると想定
model.eval()

# クラス名のリストと各クラスに対応する色を定義 (BGR形式)
class_names = ['Awamori', 'Nihonshu', 'No Alcohol', 'Shochu', 'Wine']
class_colors = {
    'Awamori': (147, 86, 219),    # 落ち着いたピンク (BGR)
    'Nihonshu': (96, 183, 99),    # 落ち着いたグリーン (BGR)
    'No Alcohol': (230, 153, 60), # 落ち着いたブルー (BGR)
    'Shochu': (112, 176, 255),    # 落ち着いたオレンジ (BGR)
    'Wine': (166, 232, 232)       # 落ち着いたイエロー (BGR)
}

# Streamlitアプリケーションのタイトル
st.title('Real-time brew')

# スタートボタンを作成
start_button = st.button('スタート')

# 画像表示のための場所を準備
image_placeholder = st.empty()

# メインのStreamlitループ
if start_button:
    video_capture = cv2.VideoCapture(0)  # 0はデフォルトのウェブカメラ
    stop_button = False
    try:
        while video_capture.isOpened():
            # ウェブカメラからフレームをキャプチャ
            ret, frame = video_capture.read()

            # フレームが正しくキャプチャされた場合
            if ret:
                # グレースケールに変換
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 顔の検出
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # 検出された顔にバウンディングボックスを描画
                for (x, y, w, h) in faces:
                    # 顔領域の切り出し
                    face = frame[y:y+h, x:x+w]

                    # モデルに入力するために顔をリサイズ
                    face_resized = cv2.resize(face, (112, 112))

                    # PIL形式に変換
                    face_pil = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))

                    # モデルのための画像前処理
                    transform = transforms.Compose([
                        transforms.Resize((112, 112)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
                    input_tensor = transform(face_pil).unsqueeze(0)

                    # モデルで推論
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        _, predicted = torch.max(outputs, 1)
                        class_index = predicted.item()
                        class_name = class_names[class_index]
                        confidence = probabilities[0][class_index].item() * 100

                    # バウンディングボックスの色をクラスに基づいて設定
                    color = class_colors[class_name]

                    # 顔の周りにバウンディングボックスを描画
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                    # クラス名と確率を含むテキスト
                    label = f"{class_name}: {confidence:.1f}%"

                    # クラス名の背景の半透明の四角を描画
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (x, y - text_height - 10), (x + text_width, y), color, cv2.FILLED)
                    alpha = 0.6  # 背景の透明度
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    # バウンディングボックスにクラス名と確率を白色で表示
                    cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                # バウンディングボックス付きのフレームをStreamlitに表示
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                image_placeholder.image(img_pil, caption='リアルタイム顔検出と分類', use_column_width=True)
            else:
                st.write("ウェブカメラから映像を取得できませんでした。")
            
            # '終了'ボタンを確認
            stop_button = st.button('終了')
            if stop_button:
                break

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

    # キャプチャを解放し、ウェブカメラを閉じる
    video_capture.release()
    # cv2.destroyAllWindows()  # ここを削除
