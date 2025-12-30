"""
Test script to verify chunking logic with different max_model_len settings.
"""
import os
os.chdir("/home/cetech/UNV_AI/maie")

from src.processors.llm.chunking import TextChunker
from src.config import settings

def test_chunking_with_model_len():
    """Test that chunk size scales with max_model_len."""
    
    # Test with different max_model_len values
    test_configs = [
        (8192, "Small model (8K context)"),
        (32768, "Default model (32K context)"),
        (131072, "Large model (128K context)"),
    ]
    
    # Generate a long test text (approx 50K tokens)
    test_text = " ".join([f"This is sentence number {i} in a very long document." for i in range(10000)])
    
    for max_len, description in test_configs:
        print(f"\n{'='*60}")
        print(f"{description} - max_model_len: {max_len}")
        print(f"{'='*60}")
        
        # Calculate chunk size (18% of max_model_len)
        chunk_size = int(max_len * 0.18)
        print(f"Chunk size: {chunk_size} tokens (~{chunk_size * 0.75} words)")
        
        # Create chunker
        chunker = TextChunker(
            tokenizer_path=settings.llm_sum.model,
            max_tokens=chunk_size
        )
        
        # Count tokens in test text
        total_tokens = chunker.count_tokens(test_text)
        print(f"Total tokens in test text: {total_tokens}")
        
        # Split into chunks
        chunks = chunker.split(test_text)
        print(f"Number of chunks: {len(chunks)}")
        
        # Verify each chunk is within limits
        for i, chunk in enumerate(chunks):
            chunk_tokens = chunker.count_tokens(chunk)
            print(f"  Chunk {i+1}: {chunk_tokens} tokens")
            assert chunk_tokens <= chunk_size, f"Chunk {i+1} exceeds max_tokens!"
        
        # Calculate reduce phase input size
        # Assume each chunk produces ~500 tokens of intermediate summary
        intermediate_summary_tokens = len(chunks) * 500
        reduce_input_tokens = intermediate_summary_tokens + 1000  # +1000 for prompt
        
        print(f"\nReduce phase estimate:")
        print(f"  Intermediate summaries: ~{intermediate_summary_tokens} tokens")
        print(f"  Total reduce input: ~{reduce_input_tokens} tokens")
        print(f"  Fits in context? {reduce_input_tokens < max_len}")
        
        if reduce_input_tokens >= max_len:
            print(f"  ⚠️  WARNING: Reduce phase may exceed context window!")
        else:
            headroom = max_len - reduce_input_tokens
            print(f"  ✓ Headroom: {headroom} tokens ({headroom/max_len*100:.1f}%)")

if __name__ == "__main__":
    test_chunking_with_model_len()
