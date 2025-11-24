try:
    from vllm.sampling_params import StructuredOutputsParams

    print("StructuredOutputsParams: FOUND")
except ImportError:
    print("StructuredOutputsParams: NOT_FOUND")

# The old GuidedDecodingParams alias is deprecated and intentionally not
# checked here; prefer StructuredOutputsParams and the `structured_outputs`
# parameter instead.
