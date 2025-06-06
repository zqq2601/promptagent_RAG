'''import openai
from openai import OpenAI
import time

def call_model(prompt, input_text, model_config):
    client = OpenAI(
        api_key=model_config["api_key"],
        base_url=model_config["base_url"]
    )

    try:
        response = client.chat.completions.create(
            model=model_config["name"],
            messages=[
                {"role": "user", "content": prompt + "\n" + input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling model: {e}")
        time.sleep(2)
        return ""

def evaluate_accuracy(prompt, data, model_config):
    correct = 0
    for item in data:
        response = call_model(prompt, item["input"], model_config)
        if item["target"] in response:
            correct += 1
    return correct / len(data)'''

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
            messages=[
                {"role": "user", "content": prompt + "\n" + input_text}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling model: {e}")
        time.sleep(2)
        return ""

def extract_choice(text):
    """
    从模型返回中提取ABCD选项。匹配 'A', 'B', 'C', 'D'，如有多个只取第一个。
    """
    match = re.search(r"\b([ABCD])\b", text.strip().upper())
    if match:
        return match.group(1)
    return ""

def evaluate_accuracy(prompt, data, model_config):
    correct = 0
    for i, item in enumerate(data):
        response = call_model(prompt, item["input"], model_config)
        pred_choice = extract_choice(response)
        true_choice = item["target"].strip().upper()

        is_correct = pred_choice == true_choice
        print(f"[{i+1}] Pred: {pred_choice}, Target: {true_choice}, Result: {'1' if is_correct else '0'}")

        if is_correct:
            correct += 1
    accuracy = correct / len(data)
    return accuracy



