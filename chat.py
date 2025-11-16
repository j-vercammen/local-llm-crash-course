from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")

prompt = "Hi! The name of the capital city of India is"
prompt2 = "Hi! What is the name of the capital city of India?"


for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
print()
