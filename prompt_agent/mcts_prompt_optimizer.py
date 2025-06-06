import random
import copy
from evaluator import evaluate_accuracy

class MCTSPromptOptimizer:
    def __init__(self, cfg):
        self.init_prompt = cfg["task"]["init_prompt"]
        self.iterations = cfg["mcts"]["iterations"]
        self.expand_width = cfg["mcts"]["expand_width"]
        self.depth_limit = cfg["mcts"]["depth_limit"]
        self.exploration_weight = cfg["mcts"]["exploration_weight"]
        self.model_config = cfg["model"]

    def mutate_prompt(self, prompt):
        mutations = [
            "Please carefully analyze the question below before choosing an answer.",
            "Think step-by-step to select the correct answer.",
            "Read the following question and provide the best answer.",
            "Evaluate all options and give your final answer."
        ]
        return prompt + " " + random.choice(mutations)

    def search(self, eval_data):
        prompt_log = []
        best_prompt = self.init_prompt
        best_score = evaluate_accuracy(best_prompt, eval_data, self.model_config)

        for i in range(self.iterations):
            candidates = [self.mutate_prompt(best_prompt) for _ in range(self.expand_width)]
            scores = []

            for p in candidates:
                acc = evaluate_accuracy(p, eval_data, self.model_config)
                scores.append((p, acc))
                prompt_log.append({
                    "iteration": i,
                    "prompt": p,
                    "accuracy": acc
                })

            # 更新最佳
            top = max(scores, key=lambda x: x[1])
            if top[1] > best_score:
                best_prompt, best_score = top

        return best_prompt, prompt_log



