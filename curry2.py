from icrawler.builtin import BingImageCrawler

def download_images(keyword, folder_name, max_num=50):
    crawler = BingImageCrawler(storage={'root_dir': f'images6/{folder_name}'})
    crawler.crawl(keyword=keyword, max_num=max_num)

# ダウンロード対象一覧
targets = [
    ("札幌 スープカレー チキン 野菜", "benbera"),
    ("札幌 スープカレー 海鮮", "purupuru"),
    ("札幌 スープカレー ポーク", "hooddog"),
    ("札幌 スープカレー 野菜", "spicepot"),
    ("札幌 スープカレー  Chicken", "kitaro"),
    ("札幌 スープカレー  辛い", "mill"),
    ("札幌 スープカレー  グリーン", "gop")

]

# ループ処理
for keyword, folder in targets:
    download_images(keyword, folder, 50)