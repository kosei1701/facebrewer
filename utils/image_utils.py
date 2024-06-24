# utils/image_utils.py

import cv2

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
