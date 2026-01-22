import os
from pathlib import Path
from transformers import AutoTokenizer

def download_tokenizer():
    # Define model and cache directory
    model_name = "Qwen/Qwen3-4B"
    cache_dir = Path("./data/tokenizers")
    
    print(f"Downloading tokenizer for {model_name} to {cache_dir}...")
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(cache_dir),
            trust_remote_code=True
        )
        print(f"Successfully downloaded tokenizer to {cache_dir}")
        
        # List files in cache directory to verify
        print("\nFiles in cache directory:")
        for f in cache_dir.rglob("*"):
            if f.is_file():
                print(f" - {f.relative_to(cache_dir)}")
                
    except Exception as e:
        print(f"Error downloading tokenizer: {e}")

if __name__ == "__main__":
    download_tokenizer()
