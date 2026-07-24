import re

def normalize_whitespace(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()