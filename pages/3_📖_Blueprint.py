import streamlit as st
import os

# ディレクトリのベースパスを取得
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ページ設定
st.set_page_config(
    page_icon=os.path.join(BASE_DIR, 'image', 'favicon3.png')  # ファビコンのパスを設定
)

# サイドバーに "Blueprint" 
st.sidebar.write("Blueprint")

# メインパネル
st.title('Blueprint')


 # 画像を表示
st.write("#### Application workflow")
image_path = os.path.join('image', 'blueprint1.png')
st.image(image_path, use_column_width=True)

st.write("")
st.divider()

 # 画像を表示
st.write("#### Model background")
image_path = os.path.join('image', 'blueprint2.png')
st.image(image_path, use_column_width=True)

st.write("")
st.divider()

 # 画像を表示
st.write("#### ResNet18 arquitecture")
image_path = os.path.join('image', 'blueprint3.png')
st.image(image_path, use_column_width=True)