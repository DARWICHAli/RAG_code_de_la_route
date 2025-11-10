"""
Safety filters for RAG Code de la Route
---------------------------------------
This module provides:
1. is_safe_question: detect if a question is related to the Code de la Route
2. sanitize_response: clean or truncate generated responses
"""

import re

# === Configuration ===
# List of keywords considered relevant
SAFE_KEYWORDS = [
    "permis", "conduite", "véhicule", "points", "accident",
    "circulation", "infractions", "règles", "sécurité routière",
    "contrôle", "code de la route", "amende", "signalisation","piéton","passage",
    "Stationnement","Stationner","agglomération","priorité","cédez le passage",
    "limitations de vitesse","alcoolémie","drogue au volant","ceinture de sécurité",
    "téléphone au volant","feux de signalisation","rond-point","autoroute","voie",
    "distance de sécurité","clignotant","angle mort","freinage","assurance",
    "visibilité","conditions météorologiques","équipements obligatoires",
    "contrôle technique","permis probatoire","conduite accompagnée","éconduite",
    "transport de marchandises","transport de personnes","vélo","moto","cycliste",
    "piéton","passager","usager","infrastructure routière","zone 30","zone piétonne",
    "voie rapide","voie express","péage","station-service","aire de repos",
    "vitesse","autorisée","maximale"
]

# Optional: minimum fraction of keywords required to accept question
MIN_KEYWORD_RATIO = 0.1


def is_safe_question(question: str) -> bool:
    """
    Check if a question is related to the Code de la Route.
    Returns True if safe, False otherwise.
    """
    question_lower = question.lower()
    total_keywords = len(SAFE_KEYWORDS)
    matched = sum(1 for kw in SAFE_KEYWORDS if kw in question_lower)
    ratio = matched / total_keywords
    print(ratio)
    
    return ratio >= MIN_KEYWORD_RATIO


def sanitize_response(response: str, max_length: int = 2000) -> str:
    """
    Clean generated response.
    - Truncate if too long
    - Remove excessive line breaks or empty spaces
    """
    # Remove multiple newlines
    cleaned = re.sub(r"\n+", "\n", response.strip())
    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."
    return cleaned


# === Example usage ===
if __name__ == "__main__":
    questions = [
        "Quelles sont les règles concernant le permis à points ?",
        "Quel temps fera-t-il demain à Paris ?",
        "Comment fonctionne la signalisation sur autoroute ?"
    ]

    for q in questions:
        safe = is_safe_question(q)
        print(f"Question: {q}\nSafe: {safe}\n")

    resp = "Réponse générée...\n\n\nFin de réponse."
    print("Sanitized response:", sanitize_response(resp))
