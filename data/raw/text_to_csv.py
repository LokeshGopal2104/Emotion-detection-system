import pandas as pd

# Emotion mapping (must match the dataset documentation)
emotions = [
    "joy",
    "fear",
    "anger",
    "sadness",
    "disgust",
    "shame",
    "guilt"
]

rows = []

# Read the original dataset
with open("text.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        # Find the end of the one-hot label
        end = line.find("]")

        # Extract one-hot label
        one_hot_string = line[1:end].strip()

        # Convert to list of integers
        one_hot = [int(float(x)) for x in one_hot_string.split()]

        # Extract text
        text = line[end + 1:].strip()

        # Convert one-hot vector to emotion label
        emotion_label = ", ".join(
            emotions[i] for i, value in enumerate(one_hot) if value == 1
        )

        # Store the row
        rows.append({
            "Text": text,
            "One_Hot_Label": one_hot,
            "Emotion_Label": emotion_label
        })

# Create DataFrame
df = pd.DataFrame(rows)

# Display first few rows
print(df.head())

# Save as CSV
df.to_csv("emotion_dataset.csv", index=False)

print("Dataset saved successfully as 'emotion_dataset.csv'")