from datasets import load_dataset
import pandas as pd
import os

def create_cve_knowledge_base():
    # 加载数据集并选择训练集分割
    dataset = load_dataset('rkreddyp/cve-all')
    
    # 检查数据集结构并选择正确的分割
    if isinstance(dataset, dict):  # 如果是DatasetDict
        if 'train' in dataset:
            dataset = dataset['train']
        else:
            # 如果没有'train'分割，选择第一个可用分割
            dataset = dataset[next(iter(dataset.keys()))]
    
    # 转换为Pandas DataFrame并选择前500条
    df = dataset.to_pandas().head(500)
    
    # 规范列名提取（严格匹配大小写和格式）
    required_columns = ['CVE_ID', 'Description']
    
    # 检查列是否存在
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据集缺少必要列: {missing_cols}")
    
    # 提取所需列并清理数据
    cve_data = df[required_columns].copy()
    
    # 删除无效描述行（包括空描述和"No description available"）
    cve_data = cve_data[
        (cve_data['Description'].notna()) & 
        (cve_data['Description'] != 'No description available') & 
        (cve_data['Description'] != '')
    ]
    
    # 重置索引
    cve_data = cve_data.reset_index(drop=True)
    
    # 创建知识库条目
    cve_data['knowledge_entry'] = cve_data.apply(
        lambda row: f"{row['CVE_ID']}, {row['Description']}".replace('\n', ' ').strip(), 
        axis=1
    )
    
    # 指定保存路径
    save_path = "D:/A-project/prompt/v1-prompt_agent/datasets/cve.txt"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存到文本文件
    with open(save_path, 'w', encoding='utf-8') as f:
        for entry in cve_data['knowledge_entry']:
            f.write(entry + '\n\n')
    
    print(f"成功生成CVE知识库，包含{len(cve_data)}条有效记录")
    print(f"已过滤掉{500 - len(cve_data)}条无效记录")
    print(f"文件已保存至: {save_path}")

if __name__ == "__main__":
    create_cve_knowledge_base()