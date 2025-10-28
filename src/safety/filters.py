# src/safety/filters.py
import re

BANNED = [re.compile(r"terror|explos", re.I)]

def check_input(text):
    if any(p.search(text) for p in BANNED):
        return False, "banned"
    if len(text) < 3:
        return False, "too_short"
    return True, None

def should_decline(question, retrieved):
    # if retrieved top score < threshold -> decline
    return retrieved[0]["score"] < 0.2
