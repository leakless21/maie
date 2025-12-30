import logging
from typing import List, Optional
import spacy
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class TextChunker:
    """
    Split text into chunks suitable for LLM processing.
    Uses spacy for sentence-based splitting and falls back to fixed-token chunks
    if punctuation is missing or sentences are too long.
    """
    
    def __init__(self, tokenizer_path: str, max_tokens: int = 6000, overlap_tokens: int = 200):
        """
        Initialize the chunker.
        
        Args:
            tokenizer_path: Path to the model/tokenizer for token counting.
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Number of tokens to overlap between chunks (for fallback).
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for TextChunker from {tokenizer_path}: {e}. Will use character-based estimation.")
            self.tokenizer = None
            
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._nlp = None
    
    def _get_nlp(self):
        """Lazy load spacy model."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load("xx_sent_ud_sm")
            except Exception as e:
                logger.warning(f"Failed to load spacy model xx_sent_ud_sm: {e}. Falling back to basic splitting.")
                self._nlp = None
        return self._nlp

    def count_tokens(self, text: str) -> int:
        """Count tokens in text with character-based fallback."""
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception as e:
                logger.warning(f"Tokenizer encoding failed in TextChunker: {e}")
        
        # Fallback: 3.5 chars per token for multilingual support
        return int(len(text) / 3.5)

    def split(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        1. Try sentence-based splitting using spacy.
        2. If text has no punctuation or sentences are too long, fallback to fixed-token chunks.
        """
        if not text:
            return []

        # Check if text has punctuation
        has_punctuation = any(c in text for c in ".!?。！？")
        
        nlp = self._get_nlp()
        if nlp and has_punctuation:
            return self._split_by_sentences(text)
        else:
            return self._split_fixed_tokens(text)

    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text by sentences, grouping them into chunks."""
        nlp = self._get_nlp()
        doc = nlp(text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if not sent_text:
                continue
                
            sent_tokens = self.count_tokens(sent_text)
            
            # If a single sentence is longer than max_tokens, split it by fixed tokens
            if sent_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence
                sub_chunks = self._split_fixed_tokens(sent_text)
                chunks.extend(sub_chunks)
                continue

            if current_tokens + sent_tokens > self.max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent_text]
                current_tokens = sent_tokens
            else:
                current_chunk.append(sent_text)
                current_tokens += sent_tokens
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def _split_fixed_tokens(self, text: str) -> List[str]:
        """Fallback: split text into fixed-token chunks with overlap."""
        if self.tokenizer is not None:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                chunks = []
                
                start = 0
                while start < len(tokens):
                    end = min(start + self.max_tokens, len(tokens))
                    chunk_tokens = tokens[start:end]
                    chunks.append(self.tokenizer.decode(chunk_tokens, skip_special_tokens=True))
                    
                    if end == len(tokens):
                        break
                        
                    start = end - self.overlap_tokens
                    if start < 0:
                        start = 0
                    if start >= end:
                        start = end
                        
                return chunks
            except Exception as e:
                logger.warning(f"Fixed-token splitting failed: {e}. Falling back to character-based splitting.")

        # Character-based fallback
        # max_tokens * 3.5 chars per chunk
        max_chars = int(self.max_tokens * 3.5)
        overlap_chars = int(self.overlap_tokens * 3.5)
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            
            if end == len(text):
                break
                
            start = end - overlap_chars
            if start < 0:
                start = 0
            if start >= end:
                start = end
                
        return chunks
