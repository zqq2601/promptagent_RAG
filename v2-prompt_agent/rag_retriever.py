from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import time

if not hasattr(torch.backends, 'mps'):
    class DummyMPS:
        def is_available(self): return False
        def is_built(self): return False
    torch.backends.mps = DummyMPS()

class RAGRetriever:
    def __init__(self, kb_dict, top_k=1, recall_size=100):
        """
        kb_dict: dict，格式如 {"Relevant knowledge": "path/to/kb.txt", "Vulnerability description": "path/to/CVE.txt"}
        recall_size: 粗召回阶段保留的候选文档数量
        """
        self.kb_sections = {}
        self.embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
        self.doc_embeddings = {}
        self.top_k = top_k
        self.recall_size = recall_size  # 粗召回候选数
        self.tfidf_vectorizers = {}     # TF-IDF向量器
        self.tfidf_matrices = {}        # TF-IDF文档矩阵

        # 加载知识库并构建索引
        for section, path in kb_dict.items():
            with open(path, "r", encoding="utf-8") as f:
                docs = [line.strip() for line in f if line.strip()]
                
            if not docs:  # 跳过空知识库
                continue
                
            self.kb_sections[section] = docs
            
            # 构建TF-IDF索引用于粗召回
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4))
            tfidf_matrix = vectorizer.fit_transform(docs)
            self.tfidf_vectorizers[section] = vectorizer
            self.tfidf_matrices[section] = tfidf_matrix
            
            # 预计算稠密嵌入用于精排序（仅当文档数量>召回大小时需要）
            if len(docs) > self.recall_size:
                self.doc_embeddings[section] = self.embedding_model.encode(
                    docs, batch_size=32, show_progress_bar=False, normalize_embeddings=True
                )

    def _tfidf_retrieve(self, query, section):
        """粗召回阶段：使用TF-IDF快速筛选候选文档"""
        vectorizer = self.tfidf_vectorizers[section]
        tfidf_matrix = self.tfidf_matrices[section]
        
        # 将查询转换为TF-IDF向量
        query_vec = vectorizer.transform([query])
        
        # 计算余弦相似度并获取候选索引
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        candidate_indices = scores.argsort()[-self.recall_size:][::-1]
        
        # 返回候选文档和原始相似度分数
        return candidate_indices, scores[candidate_indices]

    def retrieve(self, query):
        """优化后的检索接口（保持原接口不变）"""
        result = {}
        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        )[0]

        for section, docs in self.kb_sections.items():
            n_docs = len(docs)
            
            # 小知识库直接使用精确检索
            if n_docs <= self.recall_size:
                scores = cosine_similarity(
                    [query_embedding], self.doc_embeddings.get(section, None) or
                    self.embedding_model.encode(docs, normalize_embeddings=True)
                ).flatten()
                top_indices = scores.argsort()[-self.top_k:][::-1]
                result[section] = [docs[i] for i in top_indices]
                continue

            # 两阶段检索流程
            # 阶段1：TF-IDF粗召回
            candidate_indices, _ = self._tfidf_retrieve(query, section)
            candidate_docs = [docs[i] for i in candidate_indices]
            
            # 阶段2：稠密嵌入精排序
            candidate_embeddings = (
                self.doc_embeddings[section][candidate_indices] 
                if section in self.doc_embeddings else
                self.embedding_model.encode(candidate_docs, normalize_embeddings=True)
            )
            
            scores = cosine_similarity([query_embedding], candidate_embeddings).flatten()
            top_in_candidate = scores.argsort()[-self.top_k:][::-1]
            
            # 映射回原始索引
            final_indices = [candidate_indices[i] for i in top_in_candidate]
            result[section] = [docs[i] for i in final_indices]
            
        return result
