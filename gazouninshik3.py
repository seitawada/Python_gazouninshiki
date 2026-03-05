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
    "okuzi": "スープカレー奥芝商店",
    "garaku": "スープカレーGARAKU（ガラク）",
    "ramai": "ラマイ スープカレー",
    "yellow": "スープカリー イエロー",
    "rojiura": "スープカレーRojiura Curry SAMURAI.",
    "rakkyo": "札幌スープカレーらっきょ",
    "kokoro": "カレー食堂 心",
    "treasure": "スープカレー TREASURE",
    "pikanti": "スープカリィ ピカンティ",
    "magic_spice": "スープカレー マジックスパイス",
    "suage_plus": "スープカレー スアゲプラス",
    "suage2": "Soup Curry Suage2",
    "chutta": "札幌スープカレー専門店 soup curry shop CHUTTA!",
    "dominica": "スープカリー専門店 札幌ドミニカ",
    "es": "カレーショップ エス",
    "ajanta": "札幌スープカレー アジャンタ総本家",
    "shabazou": "スープカレー しゃば蔵",
    "maruyama": "【カレー専門店】円山教授。スープカレー",
    "bagubagu": "札幌スープカリー ばぐばぐ",
    "besu": "アジアンスープカリー べす",
    "king": "スープカリーキング",
    "kankun": "スープカレー カンクーン",
    "spice_box": "スープカリー工房 スパイス・ボックス 札幌",
    "rakkyo_circus": "らっきょ大サーカス スープカレー",
    "suage_tenjin": "Soup Curry Suage 天神",
    "neebies": "スープカレーネイビーズ",
    "kohiihau_suage": "札幌 こうひいはうす スープカレー",
    "crazy_spice": "Hokkaido Soup Curry クレイジースパイス",
    "medicine_man": "メディスンマン 札幌 スープカレー",
    "hige_nousaku": "スープカリーひげ男爵",
    "soupcurry_34": "スープカレー店34",
    "ganesha": "スープカレー＆インド料理 ガネシャ（GANESHA）",
    "sama": "Curry&Café SAMA スープカレー",
    "gogodou": "カリー乃五〇堂",
    "zora": "スープカレーZORA",
    "delhi": "デリー 札幌店 （DELHI）スープカレー",
    "savoy": "curry SAVOY スープカレー",
    "kouya": "すーぷかりー香屋 札幌",
    "teracotta": "カレー屋テラコッタ スープカレー",
    "bambi": "Bambi（バンビ）札幌 スープカレー",
    "higuma": "札幌スープカレーHIGUMA（ヒグマ）",
    "hiri": "スープカレーヒリヒリ2号",
    "tenjiku": "スープカレー天竺",
    "suriya": "スーリヤ藤野",
    "itou": "札幌スープカレー専門店エスパーイトウ",
    "ezon": "札幌スープカレー 蝦夷夢 えぞん",
    "benbera": "ベンベラネットワークカンパニー",
    "purupuru": "村上カレー店・プルプル",
    "hooddog": "スープカレー Hot Spice Shop Hood Dog",
    "spicepot": "スープカレーカリーキッチン スパイスポット! SPICE POT!",
    "kitaro": "スープカレー 木多郎 澄川本店",
    "spice_mill": "Spice&mill 札幌 スープカレー",
    "gop": "gop(ゴップ)のアナグラ スープカレー"
}

jp_result = jp_labels.get(label_en, label_en)

print("予測結果:", jp_result)
print("信頼度:", f"{confidence:.2f}%")