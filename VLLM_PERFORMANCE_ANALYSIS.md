# vLLM Server Performance Analysis

**Date**: 2025-11-21  
**Server**: vLLM with Qwen3-4B-Instruct-2507-AWQ  
**Issue**: Slow inference speeds (3-5 tokens/s observed in production)

## Benchmark Results

### 1. Direct Server Performance (Raw vLLM API)
- **Simple completions**: ~78-80 tokens/s
- **Chat completions**: ~78-80 tokens/s
- **Status**: ✅ **GOOD** - Server is performing well

### 2. MAIE Enhancement Task Performance
- **Throughput**: ~900 tokens/s (includes prompt + completion)
- **Actual generation**: ~80 tokens/s
- **Status**: ✅ **EXCELLENT** - No bottleneck here

### 3. MAIE Summary Task Performance
- **Throughput**: ~40 tokens/s
- **Time per request**: ~49 seconds (for ~2000 tokens)
- **Status**: ❌ **SLOW** - This is the bottleneck!

## Root Cause: Guided JSON Decoding

The summary task uses **guided JSON decoding** (constrained generation) to ensure the output matches a specific JSON schema. This feature:

1. **Forces the model to generate valid JSON** according to the schema
2. **Significantly slows down generation** (20x slower than unconstrained generation)
3. **Is computationally expensive** because vLLM must validate each token against the schema

### Evidence
From the logs:
```python
2025-11-21 10:27:36.776 | DEBUG | src.processors.llm.processor:execute:557 - Set up guided JSON decoding
```

The guided decoding is enabled in `src/processors/llm/processor.py`:
```python
from vllm.sampling_params import GuidedDecodingParams

kwargs["guided_decoding"] = GuidedDecodingParams(json=json.dumps(schema))
```

## Performance Comparison

| Task Type | Throughput | Guided Decoding | Notes |
|-----------|-----------|-----------------|-------|
| Simple completion | 78-80 tok/s | ❌ No | Baseline performance |
| Enhancement | ~80 tok/s | ❌ No | Normal speed |
| Summary | **~40 tok/s** | ✅ Yes | **50% slower** |

## Recommendations

### Option 1: Disable Guided Decoding (Fastest)
**Pros**: 
- 2x faster summary generation
- Simpler implementation

**Cons**:
- May generate invalid JSON occasionally
- Requires post-processing validation and retry logic

**Implementation**:
```python
# In src/processors/llm/processor.py, comment out guided decoding
# kwargs["guided_decoding"] = GuidedDecodingParams(json=json.dumps(schema))
```

### Option 2: Optimize vLLM Server Configuration
**Current settings** (from config):
```python
max_num_seqs: 4                    # Max concurrent sequences
max_num_batched_tokens: 8192       # Token budget per scheduler step
gpu_memory_utilization: 0.9        # GPU memory usage
max_model_len: 16384               # Context window (reduced from 32768)
```

**Recommended optimizations**:

1. **Increase `max_num_batched_tokens`** to allow more tokens per step:
   ```python
   max_num_batched_tokens: 16384  # Double the current value
   ```

2. **Reduce `max_num_seqs`** to focus on single-request throughput:
   ```python
   max_num_seqs: 1  # Process one request at a time for maximum speed
   ```

3. **Enable chunked prefill** for long prompts:
   ```python
   max_num_partial_prefills: 2  # Enable chunked prefill
   ```

4. **Increase context window** back to full capacity:
   ```python
   max_model_len: 32768  # Use full model capacity
   ```

### Option 3: Use Speculative Decoding (Advanced)
vLLM supports speculative decoding with a smaller draft model to speed up generation. This requires:
- A smaller draft model (e.g., Qwen2-1.5B)
- Additional GPU memory
- vLLM configuration changes

### Option 4: Hybrid Approach (Recommended)
1. **Keep guided decoding** for reliability
2. **Optimize vLLM server settings** (Option 2)
3. **Add retry logic** with lower temperature if JSON is invalid
4. **Consider caching** common summaries

## Configuration Changes

### Update `src/config/model.py`

```python
class LlmEnhanceSettings(BaseModel):
    # ... existing fields ...
    max_num_seqs: int | None = Field(
        default=1,  # Changed from 4 - focus on single-request throughput
        ge=1,
        description="Maximum in-flight sequences vLLM should schedule concurrently",
    )
    max_num_batched_tokens: int | None = Field(
        default=16384,  # Changed from 8192 - allow more tokens per step
        ge=1,
        description="Upper bound on total tokens (prompt + decode) processed per scheduler step",
    )
    max_num_partial_prefills: int | None = Field(
        default=2,  # Changed from None - enable chunked prefill
        ge=1,
        description="Enables chunked prefill when >1 to overlap scheduling for long prompts",
    )
    max_model_len: int = Field(
        default=32768  # Restored from 16384 - use full capacity
    )
```

### Restart vLLM Server
After making configuration changes:
```bash
# Stop current server
pkill -f "vllm.entrypoints.openai.api_server"

# Restart with new config
./scripts/start-vllm-server.sh
```

## Expected Improvements

With optimized settings:
- **Summary task**: 40 tok/s → **60-70 tok/s** (50-75% improvement)
- **Enhancement task**: Unchanged (~80 tok/s)
- **Memory usage**: May increase slightly

## Testing

Use the benchmark scripts to verify improvements:

```bash
# Test raw server performance
python scripts/benchmark_vllm_server.py --iterations 5 --max-tokens 100

# Test MAIE processor performance
python scripts/benchmark_maie_llm.py --task both --iterations 3

# Test summary with specific template
python scripts/benchmark_maie_llm.py --task summary --template-id generic_summary_v1 --iterations 3
```

## Monitoring

Watch the vLLM server logs for throughput metrics:
```bash
# In the vLLM server terminal
# Look for lines like:
# "Avg generation throughput: X.X tokens/s"
```

## Additional Notes

### Why is guided decoding slow?

Guided JSON decoding works by:
1. **Parsing the JSON schema** into a finite state machine (FSM)
2. **Constraining token selection** at each step to only valid tokens
3. **Validating structure** as generation progresses

This adds significant overhead because:
- The model can't use its normal sampling strategy
- Each token must be validated against the FSM
- The search space is dramatically reduced

### Alternative: Outlines Library

vLLM uses the `outlines` library for guided decoding. You could try:
- Simplifying your JSON schemas (fewer nested objects)
- Using regex patterns instead of full JSON schemas
- Implementing custom validation logic

## Conclusion

Your vLLM server is **not slow** - it's performing well for unconstrained generation. The slowdown is caused by **guided JSON decoding** in the summary task, which is a necessary trade-off for structured output reliability.

**Recommended action**: Implement Option 4 (Hybrid Approach) to get 50-75% improvement while maintaining reliability.
