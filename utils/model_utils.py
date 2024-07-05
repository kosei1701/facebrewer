import torch
import torchvision.models as models

def load_model(model_path, num_classes=5):
    # ResNet18をロードし、最終層を再定義
    model_ft = models.resnet18(pretrained=False)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    # モデルのパラメータを読み込み
    model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model_ft
