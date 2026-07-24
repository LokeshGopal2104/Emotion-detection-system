import re

def normalize_numbers(text):
    text = re.sub(
        r"\b\d+(?:\.\d+)?(?:st|nd|rd|th)?\b",
        " NUM ",
        text
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()