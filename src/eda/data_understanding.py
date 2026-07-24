import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

plt.style.use("ggplot")

# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv("data/processed_data/emotion_dataset.csv")

print("=" * 70)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 70)

# ==========================================================
# Basic Inspection
# ==========================================================

print("\nFIRST 5 ROWS")
print(df.head())

print("\nLAST 5 ROWS")
print(df.tail())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATASET INFORMATION")
df.info()

# ==========================================================
# Missing Values
# ==========================================================

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nMISSING VALUE PERCENTAGE")
print((df.isnull().sum() / len(df)) * 100)

# ==========================================================
# Duplicate Analysis
# ==========================================================

duplicates = df.duplicated().sum()

print("\nDUPLICATE ROWS :", duplicates)

if duplicates > 0:
    print("\nDuplicate Rows")
    print(df[df.duplicated()])

# Display all duplicate rows
duplicate_rows = df[df.duplicated(keep=False)]

print("\nTotal Duplicate Rows :", len(duplicate_rows))

if len(duplicate_rows) > 0:
    print(duplicate_rows.sort_values(by="Text").head(20))

# ==========================================================
# Conflicting Labels
# ==========================================================

conflicting = (
    df.groupby("Text")["Emotion_Label"]
      .nunique()
)

conflicting = conflicting[conflicting > 1]

print("\nTexts With Conflicting Labels :", len(conflicting))

conflicting_texts = conflicting.index

conflicting_rows = (
    df[df["Text"].isin(conflicting_texts)]
      .sort_values("Text")
)

if len(conflicting_rows) > 0:
    print("\nConflicting Rows")
    print(conflicting_rows)

# ==========================================================
# Class Distribution
# ==========================================================

print("\nCLASS DISTRIBUTION")
print(df["Emotion_Label"].value_counts())

plt.figure(figsize=(8,5))

df["Emotion_Label"].value_counts().plot(
    kind="bar",
    color="steelblue"
)

plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# ==========================================================
# Character Length
# ==========================================================

df["Character_Count"] = df["Text"].apply(len)

print("\nCHARACTER LENGTH STATISTICS")
print(df["Character_Count"].describe())

plt.figure(figsize=(8,5))

plt.hist(
    df["Character_Count"],
    bins=30
)

plt.title("Character Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ==========================================================
# Word Count
# ==========================================================

df["Word_Count"] = df["Text"].apply(
    lambda x: len(str(x).split())
)

print("\nWORD COUNT STATISTICS")
print(df["Word_Count"].describe())

plt.figure(figsize=(8,5))

plt.hist(
    df["Word_Count"],
    bins=30
)

plt.title("Word Count Distribution")
plt.xlabel("Words")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ==========================================================
# Vocabulary Analysis
# ==========================================================

all_words = []

for text in df["Text"]:
    all_words.extend(
        str(text).lower().split()
    )

vocabulary = set(all_words)

print("\nVOCABULARY SIZE :", len(vocabulary))

word_frequency = Counter(all_words)

print("\nTOP 20 MOST FREQUENT WORDS")

for word, freq in word_frequency.most_common(20):
    print(f"{word:15} {freq}")

top_words = pd.DataFrame(
    word_frequency.most_common(20),
    columns=["Word", "Frequency"]
)

plt.figure(figsize=(12,5))

plt.bar(
    top_words["Word"],
    top_words["Frequency"]
)

plt.xticks(rotation=45)

plt.title("Top 20 Frequent Words")

plt.tight_layout()

plt.show()

# ==========================================================
# Text Pattern Analysis
# ==========================================================

# Special Characters
special_pattern = r"[^A-Za-z0-9\s]"

special_rows = df["Text"].str.contains(
    special_pattern,
    regex=True
).sum()

print("\nROWS CONTAINING SPECIAL CHARACTERS :", special_rows)

# Numbers
number_pattern = r"\d"

number_rows = df["Text"].str.contains(
    number_pattern,
    regex=True
).sum()

print("ROWS CONTAINING NUMBERS :", number_rows)

# URLs
url_pattern = r"https?://\S+|www\.\S+"

url_rows = df["Text"].str.contains(
    url_pattern,
    regex=True
).sum()

print("ROWS CONTAINING URLs :", url_rows)

# Email Addresses
email_pattern = r"\S+@\S+"

email_rows = df["Text"].str.contains(
    email_pattern,
    regex=True
).sum()

print("ROWS CONTAINING EMAILS :", email_rows)

# HTML Tags
html_pattern = r"<.*?>"

html_rows = df["Text"].str.contains(
    html_pattern,
    regex=True
).sum()

print("ROWS CONTAINING HTML TAGS :", html_rows)

# ==========================================================
# Dataset Summary
# ==========================================================

print("\n")
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Total Samples          : {len(df)}")
print(f"Total Features         : {len(df.columns)}")
print(f"Missing Values         : {df.isnull().sum().sum()}")
print(f"Duplicate Samples      : {duplicates}")
print(f"Duplicate Rows         : {len(duplicate_rows)}")
print(f"Conflicting Texts      : {len(conflicting)}")
print(f"Vocabulary Size        : {len(vocabulary)}")
print(f"Special Characters     : {special_rows}")
print(f"Numbers                : {number_rows}")
print(f"URLs                   : {url_rows}")
print(f"Emails                 : {email_rows}")
print(f"HTML Tags              : {html_rows}")

print("=" * 70)

# ==========================================================
# Save Dataset
# ==========================================================

df.to_csv(
    "data/processed_data/emotion_dataset.csv",
    index=False
)

print("\nCleaned dataset saved successfully.")