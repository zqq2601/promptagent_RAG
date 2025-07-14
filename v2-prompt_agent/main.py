'''import yaml
import os
from mcts_prompt_optimizer import MCTSPromptOptimizer
from evaluator import evaluate_accuracy
from utils import load_data, save_json
from rag_retriever import RAGRetriever  # ✅ 新增导入

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

    # 提示词优化
    print("Running MCTS prompt optimization...")
    optimizer = MCTSPromptOptimizer(cfg)
    best_prompt, prompt_log = optimizer.search(eval_data)
    save_json(prompt_log, "result/prompt_search_log.json")

    # ✅ 初始化RAG检索器
    retriever = RAGRetriever(cfg["task"]["kb_path"], top_k=cfg["task"]["kb_topk"])

    # 优化提示词 + RAG 检索后的准确率
    print("Evaluating optimized prompt with RAG...")
    optimized_acc = evaluate_accuracy(
        prompt=best_prompt,
        data=test_data,
        model_config=cfg["model"],
        retriever=retriever
    )
    print(f"Optimized Prompt with RAG Accuracy: {optimized_acc:.2%}")'''

import yaml
import os
from mcts_prompt_optimizer import MCTSPromptOptimizer
from evaluator import evaluate_accuracy
from utils import load_data, save_json
from rag_retriever import RAGRetriever

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data = load_data(cfg["task"]["data_path"])
    eval_data = data[:cfg["task"]["eval_size"]]
    test_data = data[cfg["task"]["eval_size"]:cfg["task"]["eval_size"] + cfg["task"]["test_size"]]

    print("Evaluating original prompt...")
    original_acc = evaluate_accuracy(
        prompt=cfg["task"]["init_prompt"],
        data=test_data,
        model_config=cfg["model"]
    )
    print(f"Original Prompt Accuracy: {original_acc:.2%}")

    print("Running MCTS prompt optimization...")
    optimizer = MCTSPromptOptimizer(cfg)
    #best_prompt, prompt_log = optimizer.search(eval_data, original_acc)
    best_prompt, prompt_log = optimizer.search(eval_data)
    save_json(prompt_log, "result/prompt_search_log_v2.json")

    # 初始化多知识库检索器
    retriever = RAGRetriever(cfg["task"]["kb_dict"], top_k=cfg["task"]["kb_topk"])

    print("Evaluating optimized prompt with RAG...")
    optimized_acc = evaluate_accuracy(
        prompt=best_prompt,
        data=test_data,
        model_config=cfg["model"],
        retriever=retriever
    )
    print(f"Optimized Prompt with RAG Accuracy: {optimized_acc:.2%}")





