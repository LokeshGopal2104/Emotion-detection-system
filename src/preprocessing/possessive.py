import re

def remove_possessives(text):
    return re.sub(r"(\w+)'s\b", r"\1", text)