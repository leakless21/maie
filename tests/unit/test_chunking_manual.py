import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.processors.llm.chunking import TextChunker

def test_chunker():
    # Use a local model path for tokenizer if possible, or a common one
    tokenizer_path = "data/models/qwen3-4b-instruct-2507-awq"
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer path {tokenizer_path} not found. Skipping test.")
        return

    chunker = TextChunker(tokenizer_path, max_tokens=100, overlap_tokens=20)
    
    # Test with punctuation
    text_with_punct = "Đây là câu thứ nhất. Đây là câu thứ hai. Đây là câu thứ ba. " * 10
    chunks = chunker.split(text_with_punct)
    print(f"Text with punctuation: {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {len(chunk.split())} words")

    # Test without punctuation
    text_no_punct = "đây là một đoạn văn dài không có dấu câu để kiểm tra tính năng fallback của chunker " * 20
    chunks_no_punct = chunker.split(text_no_punct)
    print(f"\nText without punctuation: {len(chunks_no_punct)} chunks")
    for i, chunk in enumerate(chunks_no_punct):
        print(f"Chunk {i}: {len(chunk.split())} words")

if __name__ == "__main__":
    test_chunker()
