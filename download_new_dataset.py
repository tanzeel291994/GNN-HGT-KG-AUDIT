import json
import os
from datasets import load_dataset
from tqdm import tqdm

def prepare_musique_data(output_dir="new_datasets", train_limit=2000, eval_limit=1000):
    # 1. Load the dataset from Hugging Face
    print("Loading MuSiQue dataset from Hugging Face...")
    ds = load_dataset("dgslibisey/MuSiQue")
    
    # 2. Prepare paths
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train_samples.json")
    eval_path = os.path.join(output_dir, "eval_samples.json")
    corpus_path = os.path.join(output_dir, "musique_corpus.json")
    
    # 3. Save Train Samples
    print(f"Saving {train_limit} train samples...")
    train_data = ds['train'].select(range(min(train_limit, len(ds['train']))))
    with open(train_path, 'w') as f:
        json.dump(list(train_data), f, indent=2)
        
    # 4. Save Eval Samples
    print(f"Saving {eval_limit} eval samples...")
    # Using validation or test split for eval if available, otherwise selecting from train
    if 'validation' in ds:
        eval_data = ds['validation'].select(range(min(eval_limit, len(ds['validation']))))
    else:
        # Avoid overlap with train_data if validation split doesn't exist
        print("Validation split not found, selecting from train split (avoiding overlap)...")
        start_idx = train_limit
        end_idx = min(train_limit + eval_limit, len(ds['train']))
        eval_data = ds['train'].select(range(start_idx, end_idx))
    
    with open(eval_path, 'w') as f:
        json.dump(list(eval_data), f, indent=2)
        
    # 5. Create Corpus (Unique chunks from paragraph_text)
    print("Building corpus of unique paragraphs from selected train and eval samples...")
    corpus = []
    seen_hashes = set()
    total_paragraphs_processed = 0
    
    # Collect only from selected train and eval samples
    for subset_name, subset in [("train", train_data), ("eval", eval_data)]:
        for sample in tqdm(subset, desc=f"Processing {subset_name} for corpus"):
            for p in sample['paragraphs']:
                total_paragraphs_processed += 1
                text = p['paragraph_text']
                title = p['title']
                # Create a robust unique identifier using the full text to avoid collisions
                # We use a tuple of (title, text) which is hashable and accurate
                p_id = (title, text)
                if p_id not in seen_hashes:
                    corpus.append({
                        "title": title,
                        "text": text
                    })
                    seen_hashes.add(p_id)
                    
    print(f"Total paragraphs found: {total_paragraphs_processed}")
    print(f"Unique paragraphs saved to corpus: {len(corpus)}")
    print(f"Saving corpus to {corpus_path}...")
    with open(corpus_path, 'w') as f:
        json.dump(corpus, f, indent=2)
        
    print(f"Done! Files saved in {output_dir}/")

if __name__ == "__main__":
    prepare_musique_data()

