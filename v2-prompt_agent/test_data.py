from datasets import load_dataset
import json
import os

# 1. 加载 secqa 数据集（v2）
ds = load_dataset("zefang-liu/secqa", "secqa_v2")

def convert_all_splits(dataset_dict):
    """
    将 test 集中的样本转换为 PromptAgent 格式。
    """
    all_examples = []
    for split_name in ["test"]:
        split = dataset_dict[split_name]
        for example in split:
            question = example.get('Question', '').strip()
            opts = {
                "A": example.get('A', '').strip(),
                "B": example.get('B', '').strip(),
                "C": example.get('C', '').strip(),
                "D": example.get('D', '').strip(),
            }
            answer = example.get('Answer', '').strip().upper()

            input_text = (
                f"Question: {question}\n"
                "Options:\n"
                f"A. {opts['A']}\n"
                f"B. {opts['B']}\n"
                f"C. {opts['C']}\n"
                f"D. {opts['D']}\n"
                "Answer:"
            )

            target_scores = {k: int(k == answer) for k in opts}

            all_examples.append({
                "input": input_text,
                "target_scores": target_scores,
                "target": answer
            })
    return all_examples

# 2. 执行转换
converted_examples = convert_all_splits(ds)

# 3. 截取前100条
converted_examples = converted_examples[:20]

# 4. 保存路径
output_path = "D:/A-project/prompt/v1-prompt_agent/datasets/qa_v2_l.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# 5. 写入 JSON 文件
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"examples": converted_examples}, f, indent=2, ensure_ascii=False)

print(f"✅ 已保存 20 条 SECQA 测试样本至：{output_path}")

