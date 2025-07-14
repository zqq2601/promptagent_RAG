'''from openai import OpenAI
import time
import re

def call_model(prompt, input_text, model_config):
    client = OpenAI(
        api_key=model_config["api_key"],
        base_url=model_config["base_url"]
    )

    try:
        response = client.chat.completions.create(
            model=model_config["name"],
            messages=[{"role": "user", "content": prompt + "\n" + input_text}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling model: {e}")
        time.sleep(2)
        return ""

def extract_choice(text):
    match = re.search(r"\b([ABCD])\b", text.strip().upper())
    if match:
        return match.group(1)
    return ""

def evaluate_accuracy(prompt, data, model_config, retriever=None):
    correct = 0
    for i, item in enumerate(data):
        background = ""
        if retriever:
            background = retriever.retrieve(item["input"])
            print(f"[{i+1}] Retrieved context: {background[:80]}...")

        final_prompt = prompt + "\n" + "Context:\n" + background if background else prompt
        response = call_model(final_prompt, item["input"], model_config)
        pred_choice = extract_choice(response)
        true_choice = item["target"].strip().upper()

        is_correct = pred_choice == true_choice
        print(f"[{i+1}] Pred: {pred_choice}, Target: {true_choice}, Result: {'1' if is_correct else '0'}")

        if is_correct:
            correct += 1
    accuracy = correct / len(data)
    return accuracy'''
from openai import OpenAI
import time
import re

def call_model(prompt, input_text, model_config):
    client = OpenAI(
        api_key=model_config["api_key"],
        base_url=model_config["base_url"]
    )
    try:
        response = client.chat.completions.create(
            model=model_config["name"],
            messages=[{"role": "user", "content": prompt + "\n" + input_text}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling model: {e}")
        time.sleep(2)
        return ""

def extract_choice(text):
    match = re.search(r"\b([ABCD])\b", text.strip().upper())
    if match:
        return match.group(1)
    return ""

def evaluate_accuracy(prompt, data, model_config, retriever=None):
    correct = 0
    for i, item in enumerate(data):
        extra_context = ""
        if retriever:
            retrieved = retriever.retrieve(item["input"])
            for section_name in ["Relevant knowledge", "Vulnerability description"]:
                content = "\n".join(retrieved.get(section_name, []))
                extra_context += f"\n\n{section_name}:\n{content}"

            # 打印可解释信息
            print(f"[{i+1}] Retrieved Context:\n{extra_context.strip()[:300]}...\n")

        final_prompt = prompt + extra_context.strip()
        response = call_model(final_prompt, item["input"], model_config)

        pred_choice = extract_choice(response)
        true_choice = item["target"].strip().upper()

        is_correct = pred_choice == true_choice
        print(f"[{i+1}] Pred: {pred_choice}, Target: {true_choice}, Result: {'1' if is_correct else '0'}\n")

        if is_correct:
            correct += 1
    accuracy = correct / len(data)        
    return accuracy





