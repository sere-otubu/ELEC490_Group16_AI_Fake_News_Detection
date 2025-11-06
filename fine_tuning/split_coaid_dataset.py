import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
import re

# --- 1. Define Paths ---
# --- MODIFICATION: Set path to your CoAID folder ---
# Change this to point to the root of your CoAID dataset
COAID_DIR = '../data/CoAID/'
# ----------------------------------------------------

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

def process_coaid_data():
    """
    Loads all CoAID "News" articles, maps them to labels (0=False, 1=True),
    cleans the text, and splits them into train, validation, and test files.
    """
    
    # --- 2. Load CoAID Data ---
    print(f"Loading data from CoAID directory: {COAID_DIR}...")
    all_data = []
    
    # Find all Fake and Real news CSVs
    fake_files = glob.glob(os.path.join(COAID_DIR, '*/NewsFakeCOVID-19.csv'))
    real_files = glob.glob(os.path.join(COAID_DIR, '*/NewsRealCOVID-19.csv'))
    coaid_files = fake_files + real_files

    if not coaid_files:
        print(f"CRITICAL ERROR: No CoAID CSV files found in {COAID_DIR}. Check your path.")
        return

    coaid_added = 0
    for csv_file in coaid_files:
        try:
            # Determine label based on filename
            if 'NewsFakeCOVID-19' in csv_file:
                label = 0  # False
            else:
                label = 1  # True
                
            df_coaid = pd.read_csv(csv_file)
            
            # Use 'content' for text
            if 'title' not in df_coaid.columns or 'content' not in df_coaid.columns:
                print(f"Skipping {csv_file}: missing 'title' or 'content' column.")
                continue
                
            for index, row in df_coaid.iterrows():
                title = str(row['title'])
                text = str(row['content'])
                
                # Simple check to skip empty/invalid rows
                if title and text and len(title) > 5 and len(text) > 20:
                    all_data.append({
                        'title': title,
                        'text': text,
                        'label': label
                    })
                    coaid_added += 1
                    
        except Exception as e:
            print(f"ERROR: Could not process file {csv_file}. Reason: {e}")

    print(f"Successfully loaded {coaid_added} total articles from CoAID.")
    if not all_data:
        print("CRITICAL ERROR: No data was successfully loaded.")
        return

    # --- 3. Create DataFrame and Split ---
    print("\nCreating and splitting combined dataset...")
    df = pd.DataFrame(all_data)

    print(f"Total articles loaded: {len(df)}")
    print("Initial label distribution:")
    print(df['label'].value_counts())

    print("Cleaning all text data...")
    df['title'] = df['title'].apply(clean_text)
    df['text'] = df['text'].apply(clean_text)
    print("Text cleaning complete.")
    
    # Drop duplicates
    df.drop_duplicates(subset=['title', 'text'], inplace=True)
    print(f"Removed duplicates. {len(df)} unique articles remaining.")
    print("Final label distribution:")
    print(df['label'].value_counts())

    # Split: 80% train, 10% validation, 10% test
    try:
        train_df, remaining_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42, stratify=remaining_df['label'])
    except ValueError as e:
        print(f"WARNING: Could not stratify split. {e}. Splitting without stratification.")
        train_df, remaining_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(remaining_df, test_size=0.5, random_state=42)    

    # --- 4. Save to CSV (overwriting old files) ---
    # These will be the new files for your 2-class model
    print("\nSaving new train, validation, and test files...")
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('validation.csv', index=False)
    test_df.to_csv('test.csv', index=False)

    print("Dataset processing complete.")
    print(f"New Training set size: {len(train_df)}")
    print(f"New Validation set size: {len(val_df)}")
    print(f"New Test set size: {len(test_df)}")
    print("\n✅ Your train.csv, validation.csv, and test.csv files have been created.")

if __name__ == "__main__":
    process_coaid_data()