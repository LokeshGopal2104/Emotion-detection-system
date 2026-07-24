import re

def replace_hyphen_slash(text):
    text = re.sub(r"-", " ", text)
    text = re.sub(r"/", " ", text)
    return text


def remove_punctuation(text):
    text = re.sub(r'[.,;:()"?!&+%$÷¦]', ' ', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()