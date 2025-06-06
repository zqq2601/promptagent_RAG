# 仅一行，调用 prompt_agent
import yaml
import os
from mcts_prompt_optimizer import MCTSPromptOptimizer
from evaluator import evaluate_accuracy
from utils import load_data, save_json

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:   
        cfg = yaml.safe_load(f)

    # 加载数据
    data = load_data(cfg["task"]["data_path"])
    eval_data = data[:cfg["task"]["eval_size"]]
    test_data = data[cfg["task"]["eval_size"]:cfg["task"]["eval_size"] + cfg["task"]["test_size"]]

    # 原始提示词准确率
    print("Evaluating original prompt...")
    original_acc = evaluate_accuracy(
        prompt=cfg["task"]["init_prompt"],
        data=test_data,
        model_config=cfg["model"]
    )
    print(f"Original Prompt Accuracy: {original_acc:.2%}")

    # 优化提示词
    print("Running MCTS prompt optimization...")
    optimizer = MCTSPromptOptimizer(cfg)
    best_prompt, prompt_log = optimizer.search(eval_data)

    # 保存提示词搜索过程
    #save_json(prompt_log, "result/prompt_search_log.json")
    save_json(prompt_log, "result/prompt_search_log_2.json")

    # 新提示词准确率
    print("Evaluating optimized prompt...")
    optimized_acc = evaluate_accuracy(
        prompt=best_prompt,
        data=test_data,
        model_config=cfg["model"]
    )
    print(f"Optimized Prompt Accuracy: {optimized_acc:.2%}")




