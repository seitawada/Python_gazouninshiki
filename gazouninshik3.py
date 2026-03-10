import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ===== モデル構造を作る =====
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 53)  # クラス数を53に変更

# ===== 重み読み込み =====
checkpoint = torch.load("model5.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])
class_names = checkpoint["classes"]

model.eval()  # モデルを評価モードに設定

# 画像の読み込み
image = Image.open('image_kotae/soupcurry7.jpg').convert('RGB')  # 画像をRGBに変換
image = image.resize((224, 224))  # サイズを224x224に変更

# 画像の前処理
image_array = np.array(image).astype("float32") / 255.0  # 正規化
image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)  # テンソルに変換

#推論の実行
with torch.no_grad(): 
    output = model(image_tensor)  # モデルに画像を入力して出力を得る
    probs = F.softmax(output, dim=1)  # 出力を確率に変換
    confidence, predicted_class = torch.max(probs, dim=1)  # 最も高い確率とそのクラスを取得

predicted_class = predicted_class.item()  # クラス番号を取得
confidence = confidence.item() * 100  # 信頼度を取得

label_en = class_names[predicted_class]


# 予測結果を日本語訳
jp_labels = {
    "okuzi": "某スープカレー店A",
    "garaku": "某スープカレー店B",
    "ramai": "某スープカレー店C",
    "yellow": "某スープカレー店D",
    "rojiura": "某スープカレー店E",
    "rakkyo": "某スープカレー店F",
    "kokoro": "某スープカレー店G",
    "treasure": "某スープカレー店H",
    "pikanti": "某スープカレー店I",
    "magic_spice": "某スープカレー店J",
    "suage_plus": "某スープカレー店K",
    "suage2": "某スープカレー店L",
    "chutta": "某スープカレー店M",
    "dominica": "某スープカレー店N",
    "es": "某スープカレー店O",
    "ajanta": "某スープカレー店P",
    "shabazou": "某スープカレー店Q",
    "maruyama": "某スープカレー店R",
    "bagubagu": "某スープカレー店S",
    "besu": "某スープカレー店T",
    "king": "某スープカレー店U",
    "kankun": "某スープカレー店V",
    "spice_box": "某スープカレー店W",
    "rakkyo_circus": "某スープカレー店X",
    "suage_tenjin": "某スープカレー店Y",
    "neebies": "某スープカレー店Z",
    "kohiihau_suage": "某スープカレー店AA",
    "crazy_spice": "某スープカレー店AB",
    "medicine_man": "某スープカレー店AC",
    "hige_nousaku": "某スープカレー店AD",
    "soupcurry_34": "某スープカレー店AE",
    "ganesha": "某スープカレー店AF",
    "sama": "某スープカレー店AG",
    "gogodou": "某スープカレー店AH",
    "zora": "某スープカレー店AI",
    "delhi": "某スープカレー店AJ",
    "savoy": "某スープカレー店AK",
    "kouya": "某スープカレー店AL",
    "teracotta": "某スープカレー店AM",
    "bambi": "某スープカレー店AN",
    "higuma": "某スープカレー店AO",
    "hiri": "某スープカレー店AP",
    "tenjiku": "某スープカレー店AQ",
    "suriya": "某スープカレー店AR",
    "itou": "某スープカレー店AS",
    "ezon": "某スープカレー店AT",
    "benbera": "某スープカレー店AU",
    "purupuru": "某スープカレー店AV",
    "hooddog": "某スープカレー店AW",
    "spicepot": "某スープカレー店AX",
    "kitaro": "某スープカレー店AY",
    "spice_mill": "某スープカレー店AZ",
    "gop": "某スープカレー店BA"
}

jp_result = jp_labels.get(label_en, label_en)

print("予測結果:", jp_result)
print("信頼度:", f"{confidence:.2f}%")