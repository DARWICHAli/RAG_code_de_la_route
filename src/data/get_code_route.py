import os
import requests
import json
from pathlib import Path
from time import sleep

# Ton token Bearer
TOKEN = os.getenv("LEGIFRANCE_TOKEN")
if not TOKEN:
    raise ValueError("Token non trouvé dans .env")

# Chemin vers le fichier avec les IDs d'articles
IDS_PATH = Path("data/raw/code_route_article_ids.json")

# Chemin de sortie du JSONL
OUTPUT_PATH = Path("data/processed/code_route_articles_V0.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Lire les IDs depuis le fichier
with open(IDS_PATH, "r", encoding="utf-8") as f:
    article_ids = json.load(f)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
    for art_id in article_ids:
        try:
            resp = requests.post(
                "https://api.piste.gouv.fr/dila/legifrance/lf-engine-app/consult/getArticle",
                headers=HEADERS,
                json={"id": art_id},
            )
            resp.raise_for_status()
            data = resp.json()
            article = data.get("article", {})

            # Extraire titresTM
            titres_tm = article.get("context", {}).get("titresTM", [])
            titresTM_list = [t.get("titre") for t in titres_tm if "titre" in t]

            # Identifier le titre de la section (dernier titre TM)
            titre_section = titresTM_list[-1] if titresTM_list else None

            out = {
                "id": article.get("id"),
                "num": article.get("num"),
                "computedNums": article.get("computedNums", []),
                "texte": article.get("texte"),
                "titreSection": titre_section,
                "fullSectionsTitre": article.get("fullSectionsTitre"),
                "titresTM": titresTM_list
            }

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[INFO] Article {art_id} sauvegardé")
            sleep(0.1)  # léger délai pour éviter de saturer l'API
        except requests.HTTPError as e:
            print(f"[ERROR] Échec pour {art_id}: {e}")
        except Exception as e:
            print(f"[ERROR] Autre erreur pour {art_id}: {e}")

print(f"[INFO] JSONL complet sauvegardé dans {OUTPUT_PATH}")
