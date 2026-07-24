import pandas as pd

from .lowercase import convert_to_lowercase
from .whitespace import normalize_whitespace
from .contractions import expand_contractions
from .possessive import remove_possessives
from .punctuation import replace_hyphen_slash, remove_punctuation
from .numbers import normalize_numbers
from .punctuation import (
    replace_hyphen_slash,
    remove_punctuation
)
from .numbers import normalize_numbers


def clean_text(text):

    text = convert_to_lowercase(text)

    text = normalize_whitespace(text)

    text = expand_contractions(text)

    text = remove_possessives(text)

    text = replace_hyphen_slash(text)

    text = remove_punctuation(text)

    text = normalize_numbers(text)

    text = normalize_whitespace(text)

    return text


def preprocess_dataframe(df):

    df = df.copy()

    df["Text"] = df["Text"].apply(clean_text)

    return df