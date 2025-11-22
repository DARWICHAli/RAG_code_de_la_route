# app_streamlit.py
import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"  # ou l’URL de ton serveur Docker

st.set_page_config(page_title="RAG Code de la Route", page_icon="🚦")

st.title("RAG — Code de la Route Français 🚦")
st.write(
    "Posez une question sur le Code de la route français et obtenez une réponse sourcée directement depuis le texte officiel."
)

# Input utilisateur
question = st.text_area("Votre question :", height=100)

if st.button("Envoyer"):
    if not question.strip():
        st.warning("Veuillez saisir une question.")
    else:
        try:
            response = requests.post(API_URL, json={"question": question})
            if response.status_code == 200:
                data = response.json()
                st.subheader("Réponse :")
                st.write(data["answer"])
                if data.get("sources"):
                    st.subheader("Sources :")
                    for s in data["sources"]:
                        st.write(f"- num {s['num']}, score {s['score']:.2f}")
            else:
                st.error(f"Erreur API {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"Erreur de connexion à l'API : {e}")
