from vllm import LLM, SamplingParams

prompts = [
    "The color of the sky is"
]

llm = LLM(model="meta-llama/Llama-3.2-3B", max_model_len=512, enable_prefix_caching=True)
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")