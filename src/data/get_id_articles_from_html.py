from bs4 import BeautifulSoup
import json
from pathlib import Path

# Lire le HTML complet depuis un fichier
with open("data/raw/code_route_sommaire.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

article_ids = []

for a in soup.find_all("a", href=True):
    href = a['href']
    if "LEGIARTI" in href:
        if "?anchor=" in href:
            art_id = href.split("?anchor=")[1].split("#")[0]
        elif "#" in href:
            art_id = href.split("#")[1]
        else:
            art_id = href.split("/")[-1]
        article_ids.append(art_id)

article_ids = list(dict.fromkeys(article_ids))

path = Path("data/raw/code_route_article_ids.json")
path.parent.mkdir(parents=True, exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    json.dump(article_ids, f, ensure_ascii=False, indent=2)

print(f"[INFO] {len(article_ids)} articles extraits et sauvegardés dans {path}")

