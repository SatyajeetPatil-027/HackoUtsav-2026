import os
import re

# Input and Output folders
input_folder = r"D:\HackoUtsav-2026\transcripts"
output_folder = r"D:\HackoUtsav-2026\tokens"

os.makedirs(output_folder, exist_ok=True)

# Get all .txt transcript files
txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

def clean_stutter(text):
    # remove filler words
    text = re.sub(r"\b(uh|um|ah|er)\b", "", text, flags=re.IGNORECASE)

    # remove repeated words
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)

    # remove repeated syllables like pa-pa
    text = re.sub(r"(\w+)-\1", r"\1", text)

    return text.strip()

for file in txt_files:

    input_path = os.path.join(input_folder, file)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_tokens = []

    for line in lines:
        clean_line = clean_stutter(line)
        tokens = clean_line.split()
        cleaned_tokens.append(" ".join(tokens))

    output_file = os.path.join(
        output_folder,
        os.path.splitext(file)[0] + "_tokens.txt"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        for line in cleaned_tokens:
            f.write(line + "\n")

    print(f"Processed NLP tokens saved: {output_file}")

print("NLP processing completed ✅")
