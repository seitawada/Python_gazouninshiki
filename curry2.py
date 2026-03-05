from icrawler.builtin import BingImageCrawler

def download_images(keyword, folder_name, max_num=50):
    crawler = BingImageCrawler(storage={'root_dir': f'images6/{folder_name}'})
    crawler.crawl(keyword=keyword, max_num=max_num)

# ダウンロード対象一覧
targets = [
    ("ベンベラネットワークカンパニー スープカレー", "benbera"),
    ("村上カレー店・プルプル スープカレー", "purupuru"),
    ("スープカレー Hot Spice Shop Hood Dog", "hooddog"),
    ("スープカレーカリーキッチン スパイスポット! SPICE POT!", "spicepot"),
    ("スープカレー 木多郎 澄川本店", "kitaro"),
    ("Spice&mill 札幌 スープカレー", "spice_mill"),
    ("gop(ゴップ)のアナグラ スープカレー", "gop")

]

# ループ処理
for keyword, folder in targets:
    download_images(keyword, folder, 50)