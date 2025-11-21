try:
    from vllm.sampling_params import StructuredOutputsParams
    print('StructuredOutputsParams: FOUND')
except ImportError:
    print('StructuredOutputsParams: NOT_FOUND')

try:
    from vllm.sampling_params import GuidedDecodingParams
    print('GuidedDecodingParams: FOUND')
except ImportError:
    print('GuidedDecodingParams: NOT_FOUND')
