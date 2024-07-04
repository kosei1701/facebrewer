# utils/model_utils.py

import torch
from torchvision import models

def load_model(model_path):
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, 5)  # 6クラスに変更

    # 学習済みの重みをCPUにマップして読み込む
    model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_ft.eval()
    return model_ft
