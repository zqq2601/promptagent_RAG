import random
import math
import numpy as np
from evaluator import evaluate_accuracy

class Node:
    """MCTS树节点类"""
    def __init__(self, prompt, parent=None):
        self.prompt = prompt  # 节点对应的提示
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表
        self.visit_count = 0  # 访问次数
        self.total_value = 0.0  # 累计价值
        self.mean_value = 0.0  # 平均价值
        self.is_expanded = False  # 是否已扩展
    
    def uct_value(self, exploration_weight, total_visits):
        """计算节点的UCT值"""
        if self.visit_count == 0:
            return float('inf')  # 未访问过的节点优先探索
        
        # UCT公式: Q + c * sqrt(ln(N) / n)
        return self.mean_value + exploration_weight * math.sqrt(math.log(total_visits) / self.visit_count)
    
    def best_child(self, exploration_weight):
        """选择最佳子节点"""
        total_visits = self.visit_count
        return max(self.children, key=lambda child: child.uct_value(exploration_weight, total_visits))
    
    def expand(self, mutations):
        """扩展节点"""
        if self.is_expanded:
            return self
        
        # 生成变异提示作为子节点
        for _ in range(len(mutations)):
            mutation = random.choice(mutations)
            new_prompt = self.prompt + " " + mutation
            self.children.append(Node(new_prompt, parent=self))
        
        self.is_expanded = True
        return random.choice(self.children)  # 返回一个随机子节点
    
    def backpropagate(self, value):
        """反向传播价值"""
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node.mean_value = node.total_value / node.visit_count
            node = node.parent

class MCTSPromptOptimizer:
    def __init__(self, cfg):
        self.init_prompt = cfg["task"]["init_prompt"]
        self.iterations = cfg["mcts"]["iterations"]
        self.expand_width = cfg["mcts"]["expand_width"]
        self.depth_limit = cfg["mcts"]["depth_limit"]
        self.exploration_weight = cfg["mcts"]["exploration_weight"]
        self.model_config = cfg["model"]
        
        # 变异操作集合
        self.mutations = [
            "Please carefully analyze the question below before choosing an answer.",
            "Think step-by-step to select the correct answer.",
            "Read the following question and provide the best answer.",
            "Evaluate all options and give your final answer."
        ]
    
    def mutate_prompt(self, prompt):
        """创建提示的变异版本"""
        mutation = random.choice(self.mutations)
        return prompt + " " + mutation
    
    def simulate(self, node, eval_data):
        """模拟阶段 - 评估提示性能"""
        return evaluate_accuracy(node.prompt, eval_data, self.model_config)
    
    def search(self, eval_data):
        """基于MCTS的提示搜索"""
        # 初始化根节点
        root = Node(self.init_prompt)
        prompt_log = []
        
        # 初始评估
        init_score = evaluate_accuracy(root.prompt, eval_data, self.model_config)
        root.backpropagate(init_score)
        prompt_log.append({
            "iteration": 0,
            "prompt": root.prompt,
            "accuracy": init_score
        })
        
        # MCTS主循环
        for i in range(1, self.iterations + 1):
            node = root
            
            # 1. 选择阶段 - 从根节点向下遍历
            depth = 0
            while node.is_expanded and depth < self.depth_limit:
                node = node.best_child(self.exploration_weight)
                depth += 1
            
            # 2. 扩展阶段
            if not node.is_expanded:
                node = node.expand(self.mutations)
            
            # 3. 模拟阶段 - 评估当前节点
            value = self.simulate(node, eval_data)
            
            # 记录日志
            prompt_log.append({
                "iteration": i,
                "prompt": node.prompt,
                "accuracy": value
            })
            
            # 4. 反向传播
            node.backpropagate(value)
        
        # 在整个树中搜索最佳提示
        best_prompt, best_score = self.find_best_prompt(root)
        return best_prompt, prompt_log
    
    def find_best_prompt(self, root):
        """在整个树中搜索评估值最高的提示"""
        best_score = -1
        best_prompt = None
        nodes_to_explore = [root]
        
        while nodes_to_explore:
            node = nodes_to_explore.pop()
            if node.mean_value > best_score:
                best_score = node.mean_value
                best_prompt = node.prompt
            
            # 添加所有子节点到探索列表
            nodes_to_explore.extend(node.children)
        
        return best_prompt, best_score






