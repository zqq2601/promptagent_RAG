'''import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tqdm

class RAGRetriever:
    def __init__(self, kb_path, top_k=1):
        self.top_k = top_k
        with open(kb_path, "r", encoding="utf-8") as f:
            self.docs = [line.strip() for line in f if line.strip()]
        self.vectorizer = TfidfVectorizer().fit(self.docs)
        self.doc_vecs = self.vectorizer.transform(self.docs)

    def retrieve(self, query):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vecs).flatten()
        top_indices = scores.argsort()[-self.top_k:][::-1]
        return "\n".join([self.docs[i] for i in top_indices])'''


'''from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGRetriever:
    def __init__(self, kb_dict, top_k=1):
        """
        kb_dict: dict，格式如 {"Relevant knowledge": "path/to/kb.txt", "Vulnerability description": "path/to/CVE.txt"}
        """
        self.kb_sections = {}  # section_name -> [docs]
        self.vectorizers = {}  # section_name -> vectorizer
        self.doc_vecs = {}     # section_name -> doc embeddings
        self.top_k = top_k

        for section, path in kb_dict.items():
            with open(path, "r", encoding="utf-8") as f:
                docs = [line.strip() for line in f if line.strip()]
            self.kb_sections[section] = docs
            vectorizer = TfidfVectorizer().fit(docs)
            self.vectorizers[section] = vectorizer
            self.doc_vecs[section] = vectorizer.transform(docs)

    def retrieve(self, query):
        result = {}
        for section, docs in self.kb_sections.items():
            vec = self.vectorizers[section].transform([query])
            scores = cosine_similarity(vec, self.doc_vecs[section]).flatten()
            top_indices = scores.argsort()[-self.top_k:][::-1]
            result[section] = [docs[i] for i in top_indices]
        return result'''
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch
if not hasattr(torch.backends, 'mps'):
    class DummyMPS:
        def is_available(self): return False
        def is_built(self): return False
    torch.backends.mps = DummyMPS()

class RAGRetriever:
    def __init__(self, kb_dict, top_k=1):
        """
        kb_dict: dict，格式如 {"Relevant knowledge": "path/to/kb.txt", "Vulnerability description": "path/to/CVE.txt"}
        """
        self.kb_sections = {}  # section_name -> [docs]
        self.embedding_model = SentenceTransformer('BAAI/bge-small-zh-v1.5')  # 中文优化的小型嵌入模型
        self.doc_embeddings = {}  # section_name -> numpy矩阵[n_docs, embedding_dim]
        self.top_k = top_k

        # 加载所有知识库文档
        for section, path in kb_dict.items():
            with open(path, "r", encoding="utf-8") as f:
                docs = [line.strip() for line in f if line.strip()]
            self.kb_sections[section] = docs
            # 预计算文档嵌入（批处理提高效率）
            self.doc_embeddings[section] = self.embedding_model.encode(
                docs, 
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True  # 重要！余弦相似度需要归一化
            )

    def retrieve(self, query):
        """接口完全保持不变"""
        result = {}
        query_embedding = self.embedding_model.encode(
            [query], 
            normalize_embeddings=True
        )[0]  # 获取单条查询的嵌入
        
        for section, docs in self.kb_sections.items():
            # 计算余弦相似度（已归一化的向量直接点积）
            scores = cosine_similarity(
                [query_embedding],  # 保持2D输入格式
                self.doc_embeddings[section]
            ).flatten()
            
            # 获取top_k结果（保持原顺序）
            top_indices = scores.argsort()[-self.top_k:][::-1]
            result[section] = [docs[i] for i in top_indices]
            
        return result
