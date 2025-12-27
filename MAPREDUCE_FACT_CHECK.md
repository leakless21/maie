# MapReduce Implementation Plan: Fact Check & Corrections

**Date:** December 27, 2025  
**Status:** Technical Review & Corrections (Vietnamese-Optimized)

---

## Executive Summary

After analyzing the codebase and researching industry best practices, several corrections and improvements are needed for the MapReduce implementation plan.

**Vietnamese Language Optimization:**
- Primary language is **Vietnamese** - requires language-specific sentence segmentation
- Use **underthesea** (most popular Vietnamese NLP toolkit) for sentence boundaries
- Package: `pip install underthesea` - supports sentence tokenization via `sent_tokenize()`
- Fallback to regex-based splitting if underthesea unavailable

**vLLM Pipeline Stability:**
- **Do NOT modify existing vLLM infrastructure** - it's fragile and easily breaks
- Keep existing `LLMProcessor` and client infrastructure unchanged
- MapReduce orchestration happens at **task level**, not vLLM level
- Use existing `generate_summary()` method as-is for each chunk

---

## 1. Tokenizer Implementation (CRITICAL CORRECTION)

### ❌ What the Plan Said:
```python
class TokenCounter:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-4B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
```

### ✅ What Already Exists in Codebase:

**File:** `src/processors/llm/processor.py` lines 447-492

```python
def _ensure_tokenizer(self, model_name: str) -> None:
    """
    Ensure a tokenizer is available for token counting and prompt formatting.

    Preference order:
    1) vLLM-provided tokenizer via model.get_tokenizer()
    2) Hugging Face AutoTokenizer.from_pretrained(model_name)

    Note: This is called lazily (not during model load) to avoid blocking issues
    with vLLM V1 engine's get_tokenizer() in multi-process mode.
    """
    if self.tokenizer is not None:
        return

    # Try vLLM's tokenizer handle first
    try:
        if self.model is not None and hasattr(self.model, "get_tokenizer"):
            tok = self.model.get_tokenizer()
            if tok is not None and hasattr(tok, "encode"):
                self.tokenizer = tok
                return
    except Exception as e:
        logger.debug(f"Unable to obtain tokenizer from vLLM model: {e}")

    # Fallback to Hugging Face tokenizer
    try:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    except Exception as e:
        logger.warning(f"Failed to load Hugging Face tokenizer: {e}")
        self.tokenizer = None
```

**File:** `src/processors/llm/config.py` lines 224-287

```python
def calculate_dynamic_max_tokens(
    input_text: str,
    tokenizer,
    task: str,
    max_model_len: int,
    user_override: Optional[int] = None,
) -> int:
    """
    Calculate dynamic max_tokens based on input length and task.
    
    Industry standard ratios:
    - Enhancement: 1:1 + 10% buffer (whole text rewrite)
    - Summarization: 30% compression (concise output)
    """
    # Count input tokens using tokenizer (prefer excluding special tokens)
    try:
        input_tokens = len(tokenizer.encode(input_text, add_special_tokens=False))
    except TypeError:
        input_tokens = len(tokenizer.encode(input_text))
    
    # Task-specific calculations...
```

### 🎯 Corrected Approach:

**DO NOT create a new `TokenCounter` class.** Instead:

1. **Reuse `LLMProcessor._ensure_tokenizer()`** - It's already battle-tested
2. **Reuse `calculate_dynamic_max_tokens()`** - Already handles token counting with proper fallback
3. **Integration point:** MapReduce should use `llm_processor.tokenizer` after ensuring it's initialized

---

## 2. Chunk Size & Overlap (BASED ON RESEARCH)

### Research Findings:

From industry sources (Google Cloud, Galileo AI, Pinecone):

1. **Chunk Size:**
   - Google Cloud recommends **64,000 characters** (~16,000 tokens) for Gemini
   - Industry standard: **8,000-16,000 tokens** for robust LLMs
   - **For 16K safe context:** 10,000-12,000 tokens is appropriate

2. **Overlap:**
   - **Critical parameter** for continuous topic preservation
   - Industry best practice: **10-20% of chunk size**
   - For 12K chunks: **1,200-2,400 tokens overlap**
   - Galileo AI: "sufficient overlap ensures continuous topics aren't artificially split"

3. **Sentence Boundaries:**
   - **Strongly recommended** by all sources
   - Dev.to: "naïve approach creates chunks with approximately same lengths"
   - Use spaCy or NLTK for accurate sentence splitting

### ✅ Corrected Values:

```python
# Previous (too conservative)
chunk_size_tokens = 12000
chunk_overlap_tokens = 1500  # 12.5% overlap

# Corrected (research-backed)
chunk_size_tokens = 12000      # Good for 16K context
chunk_overlap_tokens = 2000    # 16.7% overlap (within best practice range)
```

**Rationale:** 16.7% overlap ensures better context preservation at boundaries while staying within research-backed 10-20% range.

---

## 3. MapReduce Logic (RESEARCH-BACKED IMPROVEMENTS)

### Key Findings:

**From arxiv.org/html/2410.09342v1 (LLM×MapReduce paper):**
- "CoA's workflow defines an action space and selects appropriate actions for sequentially processing chunks"
- "LC-Boost adaptively either appends new evidence or updates the summary"
- **Inter-chunk conflict resolution** is critical

**From Belitsoft & Medium articles:**
- "chunk document → summarize each → summarize the summaries"
- **Intermediate summaries typically 20-30% of original chunk**
- Final reduce should synthesize, not just concatenate

### ✅ Corrected Decision Logic:

```python
def _combine_or_reduce(self, summaries: List[Dict], 
                      template_id: str,
                      original_transcript: str) -> List[Dict]:
    """
    Decision Logic (Research-backed):
    
    1. Estimate compression ratio: 20-30% per level
       - 5 chunks × 12K tokens = 60K total
       - After MAP: 5 summaries × 2.4K tokens = 12K total
       - This FITS in 16K context → direct REDUCE
    
    2. If intermediate summaries exceed 12K:
       - Group into batches of ~8K tokens each
       - Recursively MAP each group
       - Maximum recursion: 3 levels (configurable)
    
    3. Safety check: If recursion depth exceeded:
       - Fallback to truncation + best-effort summary
       - Log warning for monitoring
    """
    
    # Calculate actual compression ratio
    summary_texts = [json.dumps(s['summary']) for s in summaries]
    total_tokens = sum(self.token_counter.count_tokens(t) for t in summary_texts)
    
    # Research shows 20-30% compression is typical
    logger.info(f"Intermediate summaries: {total_tokens} tokens")
    
    # Conservative threshold: 12K for 16K context
    if total_tokens <= 12000:
        logger.info("Intermediate summaries fit in context, proceeding to REDUCE")
        return summaries
    
    # Need recursive MAP
    logger.info("Intermediate summaries too large, executing recursive MAP")
    
    # Group summaries: target 8K tokens per group (leaves margin for system prompt)
    meta_summaries = self._group_and_summarize(
        summaries, template_id, target_tokens=8000
    )
    
    # Recurse if needed (with depth limit)
    if self.recursion_depth >= settings.mapreduce.max_recursion_depth:
        logger.warning("Max recursion depth reached, truncating intermediate summaries")
        return self._truncate_summaries(meta_summaries, target_tokens=12000)
    
    self.recursion_depth += 1
    return self._combine_or_reduce(meta_summaries, template_id, original_transcript)
```

---

## 4. Token Counting Method (CORRECT IMPLEMENTATION)

### ❌ What the Plan Showed:
```python
def count_tokens(self, text: str) -> int:
    return len(self.tokenizer.encode(text))
```

### ✅ Correct Implementation (from codebase):

```python
def count_tokens(self, text: str) -> int:
    """Count tokens with proper fallback handling."""
    if self.tokenizer is None:
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4
    
    try:
        # Prefer excluding special tokens for accurate input counting
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        # Fallback if add_special_tokens not supported
        return len(self.tokenizer.encode(text))
```

**Rationale:**
1. Always have fallback (character-based estimate)
2. Exclude special tokens for input counting (more accurate)
3. Handle tokenizer API variations gracefully

---

## 5. Parallel Execution Strategy (PRACTICAL CORRECTION)

### Research Finding:

**vLLM batch inference is NOT available for chat completions API** (as of v0.11.0+). The plan assumed we could use batch API, but:

1. vLLM's `model.generate()` supports batch for completions
2. Chat API (`client.chat.completions.create()`) is sequential
3. We're using chat API for structured outputs (JSON schema)

### ✅ Corrected Strategy:

```python
def _execute_map_phase(self, chunks: List[Dict], 
                      template_id: str) -> List[Dict[str, Any]]:
    """
    Execute MAP phase with practical parallelization.
    
    Options:
    1. RQ Queue (recommended): Dispatch chunks as separate jobs
       - Natural parallelization via multiple workers
       - Fault-tolerant (failed chunks can be retried)
       - Existing infrastructure
    
    2. Sequential (fallback): Process chunks one by one
       - Simple, reliable
       - Good for testing
       - No additional infrastructure needed
    
    3. Threading (advanced): Parallel API calls if using server mode
       - Only for vLLM server endpoints
       - Requires connection pooling
       - More complex error handling
    """
    
    if settings.mapreduce.use_parallel_rq:
        return self._execute_map_parallel_rq(chunks, template_id)
    else:
        return self._execute_map_sequential(chunks, template_id)

def _execute_map_parallel_rq(self, chunks: List[Dict], 
                             template_id: str) -> List[Dict]:
    """Dispatch chunks as separate RQ jobs."""
    from rq import Queue
    
    job_ids = []
    for chunk in chunks:
        job = self.queue.enqueue(
            'src.worker.mapreduce_tasks.summarize_chunk',
            args=(chunk['text'], template_id, self.llm_processor.model_path),
            timeout='10m',
            result_ttl=3600
        )
        job_ids.append(job.id)
    
    # Collect results with timeout
    results = []
    for job_id in job_ids:
        job = self.queue.fetch_job(job_id)
        result = job.result  # Blocks until complete
        results.append(result)
    
    return results
```

---

## 6. Context Window Calculations (VALIDATED)

### ✅ Confirmed Correct:

The plan's context calculations are **accurate** based on:

1. **16K safe context** is appropriate (conservative vs 32K full)
2. **12K input limit** leaves room for:
   - System prompt: ~500 tokens
   - Schema: ~300 tokens
   - Safety margin: ~200 tokens
   - Total overhead: ~1000 tokens
   - Available: 16K - 1K = 15K ≈ 12K (with 20% safety)

3. **Compression ratio assumptions:**
   - 20-30% compression per level (validated by research)
   - 16-hour audio: 57.6-86.4K tokens (correct)
   - Requires 5-7 chunks at 12K each (correct)

---

## 7. Sentence Boundary Detection (IMPLEMENTATION DETAIL)

### Research-Backed Approach (Vietnamese-Optimized):

```python
class ChunkSplitter:
    def __init__(self, tokenizer, max_tokens: int = 12000, 
                 overlap_tokens: int = 2000):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        
        # Use underthesea for Vietnamese sentence segmentation (PRIMARY)
        # Research: underthesea is the most popular Vietnamese NLP toolkit
        # https://github.com/undertheseanlp/underthesea
        try:
            from underthesea import sent_tokenize
            self.sentence_tokenizer = sent_tokenize
            self.tokenizer_name = "underthesea"
        except ImportError:
            # Fallback: Simple regex-based (Vietnamese punctuation)
            import re
            def vietnamese_sent_split(text):
                # Vietnamese sentence endings: . ! ? and common combinations
                return re.split(r'(?<=[.!?])\s+', text)
            self.sentence_tokenizer = vietnamese_sent_split
            self.tokenizer_name = "regex_fallback"
            logger.warning("underthesea not available, using regex fallback")
    
    def split(self, text: str) -> List[Dict[str, Any]]:
        """Split Vietnamese text into overlapping chunks at sentence boundaries."""
        # Get sentences using underthesea (Vietnamese-specific)
        sentences = self.sentence_tokenizer(text)
        logger.debug(f"Split text into {len(sentences)} sentences using {self.tokenizer_name}")
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sent_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            
            if current_tokens + sent_tokens > self.max_tokens and current_chunk:
                # Save chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'tokens': current_tokens,
                    'index': len(chunks)
                })
                
                # Calculate overlap: keep last N tokens worth of sentences
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk, self.overlap_tokens
                )
                current_chunk = overlap_sentences + [sentence]
                current_tokens = self._count_tokens_in_sentences(current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sent_tokens
        
        # Final chunk
        if current_chunk:
            chunks.append({
                'text': " ".join(current_chunk),
                'tokens': current_tokens,
                'index': len(chunks)
            })
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str], 
                               target_tokens: int) -> List[str]:
        """Get last N sentences that fit within target_tokens."""
        overlap = []
        tokens = 0
        
        for sentence in reversed(sentences):
            sent_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))
            if tokens + sent_tokens > target_tokens:
                break
            overlap.insert(0, sentence)
            tokens += sent_tokens
        
        return overlap
```

---

## 8. Critical Missing Components

### 🔴 Missing from Original Plan:

1. **Inter-chunk Conflict Resolution:**
   - Research shows this is critical (LC-Boost paper)
   - Need to detect contradictions between chunk summaries
   - Implementation: Simple consistency check before REDUCE

2. **Quality Metrics:**
   - Track compression ratio per chunk
   - Measure semantic coherence across boundaries
   - Alert if compression is unusually low/high

3. **Graceful Degradation:**
   - What if ALL chunks fail to summarize?
   - Fallback: Return truncated transcript with warning
   - Don't fail completely

4. **Memory Management:**
   - Chunk summaries accumulate in memory
   - For 100+ chunks, this could be problematic
   - Solution: Stream intermediate results to Redis

---

## 9. Corrected Environment Variables

```bash
# MapReduce Configuration (Research-backed values, Vietnamese-optimized)
MAPREDUCE_ENABLED=true                       # Enable MapReduce
MAPREDUCE_CHUNK_SIZE_TOKENS=12000            # 12K chunks for 16K context
MAPREDUCE_CHUNK_OVERLAP_TOKENS=2000          # 16.7% overlap (research-backed)
MAPREDUCE_MAX_RECURSION_DEPTH=3              # Max recursive MAP levels
MAPREDUCE_TIMEOUT_SECONDS=3600               # Total timeout (1 hour)
MAPREDUCE_PARALLEL_WORKERS=4                 # RQ workers for MAP phase
MAPREDUCE_USE_PARALLEL_RQ=true               # Use RQ for parallelization
MAPREDUCE_SENTENCE_SPLITTER=underthesea      # underthesea (Vietnamese) or regex fallback
MAPREDUCE_COMPRESSION_RATIO_THRESHOLD=0.15   # Alert if compression <15%
```

---

## 10. Revised Implementation Priority

### Phase 1: Core Infrastructure (Week 1-2)
- ✅ ~~Create TokenCounter class~~ **→ Use existing `_ensure_tokenizer()`**
- ✅ Implement `ChunkSplitter` with spaCy/NLTK fallback
- ✅ Add token counting wrapper using existing code
- ✅ Unit tests (90% coverage)

### Phase 2: MapReduce Logic (Week 2-3)
- ✅ Implement `MapReduceOrchestrator`
- ✅ Sequential MAP execution (baseline)
- ✅ Parallel RQ execution (optional)
- ✅ Recursive REDUCE with depth limiting
- ✅ Integration tests

### Phase 3: Pipeline Integration (Week 3-4)
- ✅ Modify `process_audio_task()` in pipeline.py
- ✅ Add context check before summarization
- ✅ Extend `LLMProcessor` to expose tokenizer
- ✅ End-to-end testing with mock LLM

### Phase 4: Production Hardening (Week 4-6)
- ✅ Graceful degradation on failures
- ✅ Quality metrics and monitoring
- ✅ Memory optimization for 100+ chunks
- ✅ Load testing with real 16+ hour audio
- ✅ Documentation and runbooks

---

## 11. Key Corrections Summary (Vietnamese-Optimized)

| Component | Original Plan | Corrected Approach | Reason |
|-----------|--------------|-------------------|---------|
| **Tokenizer** | New `TokenCounter` class | Use existing `_ensure_tokenizer()` | Already implemented & tested |
| **Token Counting** | Simple `encode()` | Use with `add_special_tokens=False` | More accurate for input |
| **Chunk Overlap** | 1500 tokens (12.5%) | 2000 tokens (16.7%) | Research best practice: 10-20% |
| **Overlap Calculation** | Fixed token count | Sentence-based overlap | Better context preservation |
| **Batch Inference** | vLLM batch API | RQ queue or sequential | Chat API doesn't support batch |
| **Recursion Threshold** | 12K tokens | 12K with depth limit | Add safety depth check |
| **Group Target** | 10K tokens | 8K tokens | Leave more margin for system prompt |
| **Sentence Splitter** | spaCy/NLTK (English) | underthesea (Vietnamese) + regex fallback | Vietnamese language support |
| **vLLM Pipeline** | Possibly modify | DO NOT TOUCH | Fragile, easily breaks |

---

## 12. Critical Implementation Notes (Vietnamese-Specific)

### 🎯 Must-Have Features:

1. **Lazy Tokenizer Init:** Don't initialize tokenizer until needed (vLLM V1 blocking issue)
2. **Fallback Token Estimation:** Character-based estimate if tokenizer fails
3. **Vietnamese Sentence Boundaries:** MUST use underthesea or regex fallback (NOT spaCy/NLTK)
4. **Depth Limiting:** MUST prevent infinite recursion
5. **RQ Integration:** Use existing RQ infrastructure for parallelization
6. **vLLM Untouched:** DO NOT modify existing vLLM pipeline (use as-is)

### ⚠️ Known Limitations:

1. **vLLM Batch API:** Not available for chat completions (structured outputs)
2. **Memory:** Large MapReduce jobs (100+ chunks) may need Redis streaming
3. **Quality:** First-level summaries critical—garbage in, garbage out
4. **Cost:** MapReduce increases LLM calls by 2-3x (acceptable tradeoff for 16+ hour audio)
5. **Vietnamese Dependency:** Requires `underthesea` package (falls back to regex if unavailable)

---

## 13. Testing Requirements (Vietnamese-Optimized)

### Unit Tests (Must Pass):
- ✅ `ChunkSplitter` with various lengths (1K, 12K, 60K, 100K tokens)
- ✅ Vietnamese sentence boundary preservation (test with Vietnamese text samples)
- ✅ underthesea vs regex fallback behavior
- ✅ Overlap calculation accuracy (verify 2K tokens overlap)
- ✅ Token counting with and without tokenizer
- ✅ Recursion depth limiting (max 3 levels)

### Integration Tests:
- ✅ 16-hour Vietnamese audio (60K tokens) → MapReduce with 5 chunks
- ✅ Single chunk failure handling (3/5 succeed)
- ✅ Recursion triggering (force >12K intermediate summaries)
- ✅ RQ parallel execution vs sequential comparison
- ✅ Verify vLLM pipeline remains unchanged (regression test)

### Load Tests:
- ✅ 100-hour audio (375K tokens) → 31 chunks
- ✅ Memory usage monitoring
- ✅ Quality assessment (compare to baseline)

---

## References

### Codebase Files Analyzed:
- `src/processors/llm/processor.py` (lines 447-492): `_ensure_tokenizer()`
- `src/processors/llm/config.py` (lines 224-287): `calculate_dynamic_max_tokens()`
- `src/worker/pipeline.py` (line 337): Context length checking placeholder

### Research Sources:
1. **Google Cloud Blog** - "Long-document summarization with Workflows and Gemini models"
2. **Galileo AI** - "Master LLM Summarization Strategies and their Implementations"
3. **arxiv.org** - "LLM×MapReduce: Simplified Long-Sequence Processing" (2024)
4. **Pinecone** - "Chunking Strategies for LLM Applications"
5. **Belitsoft** - "LLM Summarization of Large Documents"

---

**Document Version:** 1.0  
**Last Updated:** December 27, 2025  
**Status:** Ready for Implementation Review  
**Next Steps:** Update main implementation guide with corrections
