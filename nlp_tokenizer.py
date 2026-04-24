import os
import re
import nltk

nltk.download("punkt", quiet=True)

# Base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

input_folder = os.path.join(base_dir, "transcripts")
output_folder = os.path.join(base_dir, "tokens")

os.makedirs(output_folder, exist_ok=True)

def clean_stutter(text):
    text = re.sub(r"\b(uh|um|ah|er)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
    text = re.sub(r"(\w+)-\1", r"\1", text)
    return text.strip()

txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

for file in txt_files:
    input_path = os.path.join(input_folder, file)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 🔹 Step 1: Clean text
    cleaned_text = clean_stutter(text)

    # 🔹 Step 2: Sentence tokenization
    sentences = nltk.sent_tokenize(cleaned_text)

    output_file = os.path.join(
        output_folder,
        os.path.splitext(file)[0] + "_tokens.txt"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence.strip() + "\n")

    print(f"Processed: {output_file}")

print("NLP processing completed ✅")
