# Split the FakeHealth dataset into a BALANCED 2-CLASS (True/False) set
import pandas as pd
import json
import os
import glob
from sklearn.model_selection import train_test_split
import re

# --- 1. Define Paths ---
DATA_DIR = '../data/FakeHealth/dataset'
REVIEWS_DIR = os.path.join(DATA_DIR, 'reviews')
CONTENT_DIR = os.path.join(DATA_DIR, 'content')

# --- 2. Build Rating (Ground Truth) Lookup ---
print("Building rating lookup...")
ratings_lookup = {}
for review_file in [os.path.join(REVIEWS_DIR, 'HealthStory.json'), os.path.join(REVIEWS_DIR, 'HealthRelease.json')]:
    try:
        with open(review_file, 'r', encoding='utf-8') as f:
            reviews_data = json.load(f)
            for item in reviews_data:
                key = str(item.get('news_id'))
                rating = item.get('rating')
                if key is not None and rating is not None:
                    ratings_lookup[key] = rating
    except Exception as e:
        print(f"ERROR: Could not process file {review_file}. Reason: {e}")

print("Rating lookup built.")

# --- 3. Define Label Mapping (MODIFIED FOR 2-CLASS) ---
# 0, 1 -> 0 (False)
# 4, 5 -> 1 (True)
# 2, 3 -> None (Discard "Uncertain")
def map_rating_to_label(rating):
    if rating in [0, 1]:   
        return 0  # False
    if rating in [4, 5]:   
        return 1  # True
    return None # Discard 2 and 3

def clean_text(text):
    """
    Cleans text by fixing encoding errors, removing newlines,
    and collapsing extra whitespace.
    """
    if not isinstance(text, str):
        return text
    
    # Fix common encoding errors
    text = text.replace('â€œ', '“').replace('â€ ', '”')
    text = text.replace('â€™', "'").replace('â€˜', "‘")
    text = text.replace('â€”', '—')
    text = text.replace('â€¦', '…')
    
    # Remove newlines and carriage returns, replace with a space
    text = re.sub(r'[\n\r]+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- 4. Load Text and Combine with Labels ---
print("Loading article text from 'content' folder...")
all_data = []
content_files = glob.glob(os.path.join(CONTENT_DIR, 'HealthStory/*.json')) + \
                  glob.glob(os.path.join(CONTENT_DIR, 'HealthRelease/*.json'))

if not content_files:
    print(f"Warning: No original content files found in {CONTENT_DIR}.")

matches_found = 0
id_mismatches = 0
text_missing = 0

for content_file in content_files:
    try:
        filename = os.path.basename(content_file)
        news_id = os.path.splitext(filename)[0]
        with open(content_file, 'r', encoding='utf-8') as f:
            content_data = json.load(f)
        
        title = content_data.get('title', '')
        text = content_data.get('text')
        
        if not news_id or not text or not isinstance(text, str):
            text_missing += 1
            continue

        rating = ratings_lookup.get(news_id)
        
        if rating is None:
            id_mismatches += 1
            continue
            
        label = map_rating_to_label(rating)
        
        # This will now skip "Uncertain" articles (ratings 2, 3)
        if label is None:
            continue 

        all_data.append({ 'title': title, 'text': text, 'label': label})
        matches_found += 1

    except Exception as e:
        print(f"ERROR: Could not process file {content_file}. Reason: {e}")

print(f"Loaded {matches_found} 'True' and 'False' articles from FakeHealth.")

# --- 5. Create DataFrame, Clean, Balance, and Split ---
print("\nCreating and processing dataset...")
if not all_data:
    print("CRITICAL ERROR: No data was loaded. Exiting.")
    exit()

df = pd.DataFrame(all_data)

print("Cleaning all text data...")
df['title'] = df['title'].apply(clean_text)
df['text'] = df['text'].apply(clean_text)
print("Text cleaning complete.")

# Drop duplicates
df.drop_duplicates(subset=['title', 'text'], inplace=True)
print(f"Removed duplicates. {len(df)} unique articles remaining.")
print("Unbalanced label distribution:")
print(df['label'].value_counts())

# --- NEW: Balance the dataset by downsampling ---
print("\nBalancing dataset by downsampling...")
df_false = df[df['label'] == 0]
df_true = df[df['label'] == 1]

# Find the size of the smaller class
min_size = min(len(df_false), len(df_true))

if min_size == 0:
    print("CRITICAL ERROR: One class has 0 samples. Cannot balance. Exiting.")
    exit()

print(f"Minority class size: {min_size}")

# Sample the majority class to match the minority class
# Use sample(n=min_size) for both to shuffle them
df_false_balanced = df_false.sample(n=min_size, random_state=42)
df_true_balanced = df_true.sample(n=min_size, random_state=42)

# Combine the balanced DataFrames
df_balanced = pd.concat([df_false_balanced, df_true_balanced])

print("Balancing complete.")
print("Final balanced label distribution:")
print(df_balanced['label'].value_counts())
# ---------------------------------------------------

# Split the BALANCED DataFrame
print("\nSplitting balanced dataset...")
try:
    train_df, remaining_df = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label'])
    val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42, stratify=remaining_df['label'])
except ValueError as e:
    print(f"WARNING: Could not stratify split. {e}. Splitting without stratification.")
    train_df, remaining_df = train_test_split(df_balanced, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)    

# --- 6. Save to CSV (overwriting old files) ---
print("\nSaving new train, validation, and test files...")
train_df.to_csv('train.csv', index=False)
val_df.to_csv('validation.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Dataset processing complete.")
print(f"New Training set size: {len(train_df)} (Balanced)")
print(f"New Validation set size: {len(val_df)} (Balanced)")
print(f"New Test set size: {len(test_df)} (Balanced)")
print("Your train.csv, validation.csv, and test.csv files have been overwritten.")