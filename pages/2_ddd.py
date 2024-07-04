import streamlit as st
import cv2
from PIL import Image
import numpy as np

# タイトルを設定
st.title('Real-time Face Detection')

# OpenCVでデバイスのウェブカメラを起動
cap = cv2.VideoCapture(0)

# Haar Cascade 分類器の読み込み
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlitで表示するエリアを設定
img_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 顔の検出
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # バウンディングボックスを描画
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # OpenCVのカラーチャンネルをRGBに変換してPIL Imageに変換
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    # PIL ImageをStreamlitで表示
    img_placeholder.image(frame_pil)
    
    # 「q」を押すとループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャの解放とウィンドウのクローズ
cap.release()
cv2.destroyAllWindows()
