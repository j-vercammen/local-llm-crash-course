from typing import List
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained("zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf")
# llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf")


def get_prompt(instruction: str, history: List[str] = None) -> str:
    system = "You are an Al assistant that gives helpful answers.  You answer the questions in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history is not None:
        prompt += f"This is the conversation history: {''. join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(prompt)
    return prompt


"""
def get_prompt(instruction: str) -> str:
    system = "You are an Al assistant that gives helpful answers.  You answer the questions in a short and concise way."
    prompt = f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{instruction} [/INST]"
    return prompt
"""

history = []
question = "Which is the capital city of India?"
answer = ""


for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()
history.append(answer)

question = "And which is of the United States?"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()


"""
prompt = "Hi! The name of the capital city of India is"
prompt2 = "Hi! What is the name of the capital city of India?"

for word in llm(prompt, stream=True):
    print(word, end="", flush=True)
print()
"""
