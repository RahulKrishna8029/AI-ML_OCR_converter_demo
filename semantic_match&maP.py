import os
import json
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.neighbors import NearestNeighbors

# Set parallelism flag for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the embedding model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Function to generate embedding for a given text (key)
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean across the sequence length dimension to reduce dimensionality (batch_size, hidden_size)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Example file path for the source JSON file (adjust this path as needed)
source_json_path = "/Users/grahulkrishna/Downloads/structured_data.json"

# Load the source JSON from the file
with open(source_json_path, 'r') as f:
    source_json = json.load(f)

# Sample Target JSON template (with fewer keys)
target_json_template = {
    "insurance_name": "",
    "policy_number": "",
    "policy_start_date": "",
    "customer_service_phone": "",
    "insured_name": "",
    "insured_address": ""
}

# Function to merge and deduplicate address information
def merge_address(address_dict):
    address_lines = []
    for key in ['line1', 'line2', 'line3', 'area', 'city', 'state', 'pincode']:
        if key in address_dict:
            address_lines.append(address_dict[key])
    return ", ".join(address_lines)

# Step 2: Extract keys from both JSONs
source_keys = list(source_json.keys())
target_keys = list(target_json_template.keys())

# Step 3: Generate embeddings for keys from source and target JSONs
source_embeddings = [generate_embedding(key) for key in source_keys]  # Embeddings for source keys
target_embeddings = [generate_embedding(key) for key in target_keys]  # Embeddings for target keys

# Step 4: Define a nearest neighbor function to match keys based on embedding similarity
def nearest_neighbor(embeddings_target, embeddings_source):
    nbrs = NearestNeighbors(n_neighbors=1).fit(embeddings_source)
    distances, indices = nbrs.kneighbors(embeddings_target)
    return indices

# Step 5: Perform the nearest neighbor search
matched_indices = nearest_neighbor(target_embeddings, source_embeddings)

# Step 6: Assign values from Source JSON to the best matching keys in Target JSON
processed_keys = set()  # To avoid duplicate assignments
for idx, match in enumerate(matched_indices):
    best_matching_key_source = source_keys[match[0]]  # The best matching key from source JSON
    if best_matching_key_source not in processed_keys:  # Avoid duplicates
        if "address" in best_matching_key_source:  # Handle address specially
            if target_keys[idx] == "insured_address":
                address_info = source_json.get(best_matching_key_source, {})
                target_json_template["insured_address"] = merge_address(address_info)
        else:
            target_json_template[target_keys[idx]] = source_json.get(best_matching_key_source, "N/A")
        processed_keys.add(best_matching_key_source)  # Mark as processed

# Step 7: Save the target JSON to a file
target_json_path = "" #add your file path
with open(target_json_path, 'w') as f:
    json.dump(target_json_template, f, indent=4)

# Print the resulting Target JSON
print("Target JSON has been saved to:", target_json_path)
