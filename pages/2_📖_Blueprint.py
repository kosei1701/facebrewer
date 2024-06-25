import streamlit as st
import os

# ディレクトリのベースパスを取得
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ページ設定
st.set_page_config(
    page_icon=os.path.join(BASE_DIR, 'image', 'favicon1.png')  # ファビコンのパスを設定
)

# サイドバーに "Blueprint" 
st.sidebar.write("Blueprint")

# メインパネル
st.title('Blueprint')

 # 画像を表示
image_path = os.path.join('image', 'image1')
st.image(image_path, use_column_width=True)

st.markdown("""
##### Remarks

- XXXXXXXXXX
- XXXXXXXXXX
- XXXXXXXXXX
""")