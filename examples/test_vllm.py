from loguru import logger
from vllm import LLM


def main():
    llm = LLM(
        model="data/models/qwen3-4b-instruct-2507-awq",
        max_model_len=28192,
        gpu_memory_utilization=0.9,
    )
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    outputs = llm.generate(prompts)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        logger.info(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
