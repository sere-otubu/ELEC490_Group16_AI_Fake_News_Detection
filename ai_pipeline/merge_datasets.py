import json
import random

# Load Lies
with open("synthetic_fake_data_gemini.json", "r") as f:
    lies = json.load(f)

# Load Truths
with open("synthetic_true_data_gemini.json", "r") as f:
    truths = json.load(f)

print(f"Lies: {len(lies)}")
print(f"Truths: {len(truths)}")

# Combine
combined_data = lies + truths
random.shuffle(combined_data) # Critical: Shuffle so it doesn't learn patterns by order

# Save Final Dataset
output_filename = "synthetic_data_balanced.json"
with open(output_filename, "w") as f:
    json.dump(combined_data, f, indent=2)

print(f"Merged {len(combined_data)} examples into '{output_filename}'.")
print("You are ready to re-train!")