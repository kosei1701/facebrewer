# .\myenv\Scripts\activate
# streamlit run 🏠_Main.py

import streamlit as st
import os

# ディレクトリのベースパスを取得
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ページ設定
st.set_page_config(
    page_title="Face Brewer",
    page_icon=os.path.join(BASE_DIR, 'image', 'favicon3.png')  # ファビコンのパスを設定
)

# サイドバーに "Home" 
st.sidebar.write("Home")

st.title('Face Brewer')

 # 画像を表示
image_path = os.path.join('image', 'main1.png')
st.image(image_path, use_column_width=True)


st.write("""
Welcome to Face Brewer! This application detects "face" from an image and "brew" your favorite beverage based on your facial composition.

Face Brewer showcases deep learning based image processing techniques, particularly focusing on Grad-CAM
(Gradient-weighted Class Activation Mapping), MTCNN (Multi-task Cascaded Convolutional Networks), Resnet18 (Residual Network 18). 

### How to Use This App
Navigate through the sidebar to explore the functionalities of this application:

- **Home**: 
    - Back to the main page.

- **Brew**: 
    - Upload an image file.
    - The app will detect faces in the image by MTCNN, and the built-in Resnet18 model will predict a class 
    - You can see the original image with highlighted regions, and a probability bar chart showing the predicted classes and their corresponding probabilities.

- **Blueprint**: 
    - Delve into the architecture of the deep learning model used in this app.
    - Understand the different layers and their roles in image processing and feature extraction.
    - Learn about the training process, the dataset used, and the performance metrics.

We hope you find this application useful and informative. Enjoy your experience and feel free to provide feedback to help us improve!
""")
