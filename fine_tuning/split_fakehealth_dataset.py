# Split the FakeHealth dataset into train, validation, and test sets
import pandas as pd
import json
import os
import glob
from sklearn.model_selection import train_test_split

# --- 1. Define Paths ---
# Assumes your 'data' folder is at the root of the project
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
                # Normalize to string keys to avoid int/str mismatches
                key = str(item.get('news_id'))
                rating = item.get('rating')
                if key is not None and rating is not None:
                    ratings_lookup[key] = rating
    except Exception as e:
        print(f"ERROR: Could not process file {review_file}. Reason: {e}")

print("Rating lookup built.")
print(f"Articles with rating 0: {list(ratings_lookup.values()).count(0)}")
print(f"Articles with rating 1: {list(ratings_lookup.values()).count(1)}")
print(f"Articles with rating 2: {list(ratings_lookup.values()).count(2)}")
print(f"Articles with rating 3: {list(ratings_lookup.values()).count(3)}")
print(f"Articles with rating 4: {list(ratings_lookup.values()).count(4)}")
print(f"Articles with rating 5: {list(ratings_lookup.values()).count(5)}")
print(f"Total articles with ratings: {len(ratings_lookup)}")


# --- 3. Define Label Mapping ---
# 0, 1 -> 0 (False)
# 2, 3 -> 1 (Uncertain)
# 4, 5 -> 2 (True)
def map_rating_to_label(rating):
    if rating in [0, 1]:   return 0  # False
    if rating in [2, 3]:   return 1  # Uncertain
    if rating in [4, 5]:   return 2  # True
    return None

# --- 4. Load Text and Combine with Labels ---
print("Loading article text from 'content' folder...")
all_data = []
content_files = glob.glob(os.path.join(CONTENT_DIR, 'HealthStory/*.json')) + \
                  glob.glob(os.path.join(CONTENT_DIR, 'HealthRelease/*.json'))

if not content_files:
    print(f"Error: No content files found in {CONTENT_DIR}. Check your paths.")

# --- NEW: Debug counters ---
matches_found = 0
id_mismatches = 0
text_missing = 0
# ---------------------------

for content_file in content_files:
    try:
        filename = os.path.basename(content_file)
        news_id = os.path.splitext(filename)[0]
        with open(content_file, 'r', encoding='utf-8') as f:
            content_data = json.load(f)
        
        text = content_data.get('text')
        
        if not news_id or not text or not isinstance(text, str):
            text_missing += 1
            continue

        text = text.replace('\t', ' ').strip()
        while '\n\n' in text:
            text = text.replace('\n\n', '\n')
        
        if not text:
            text_missing += 1
            continue

        # Get the label from our lookup
        rating = ratings_lookup.get(news_id)
        
        if rating is None:
            id_mismatches += 1
            # --- NEW: Debug print to show the failing news_id ---
            if id_mismatches < 5: # Print first 5 mismatches
                print(f"  DEBUG: news_id '{news_id}' from content file not found in ratings lookup.")
            # ----------------------------------------------------
            continue
            
        label = map_rating_to_label(rating)
        
        if label is None:
            # This will skip articles with rating 0
            continue 

        all_data.append({'news_id': news_id, 'text': text, 'label': label})
        matches_found += 1

    except Exception as e:
        print(f"ERROR: Could not process file {content_file}. Reason: {e}")

# --- 5. Create DataFrame and Split ---
print("\n--- Processing Summary ---")
print(f"Content files found: {len(content_files)}")
print(f"Articles with missing text/id: {text_missing}")
print(f"Articles with news_id mismatches: {id_mismatches}")
print(f"Total articles successfully matched: {matches_found}")
print("--------------------------\n")

print("Creating and splitting dataset...")
df = pd.DataFrame(all_data)

if df.empty:
    print("CRITICAL ERROR: No data was loaded. This is likely due to the 'news_id' mismatch shown above.")
else:
    print(f"Successfully loaded {len(df)} articles with text and labels.")
    
    # --- NEW: Drop duplicates ---
    df.drop_duplicates(subset=['text'], inplace=True)
    print(f"Removed duplicates. {len(df)} unique articles remaining.")
    # ----------------------------

    # Ensure we have data for all 3 classes
    if df['label'].nunique() < 3:
        print(f"Warning: Dataset only contains {df['label'].nunique()} classes!")
        print(df['label'].value_counts())

    # Split: 80% train, 10% validation, 10% test
    try:
        train_df, remaining_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42, stratify=remaining_df['label'])
    except ValueError as e:
        print(f"WARNING: Could not stratify split. {e}. Splitting without stratification.")
        train_df, remaining_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)    

    # --- 6. Save to CSV (overwriting old files) ---
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('validation.csv', index=False)
    test_df.to_csv('test.csv', index=False)

    print("Dataset processing complete.")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")