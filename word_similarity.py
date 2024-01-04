# -*- coding: utf-8 -*-

import os
import numpy as np
from loguru import logger
import torch
from typing import List, Union, Dict

# 获取当前文件的绝对路径
pwd_path = os.path.abspath(os.path.dirname(__file__))
# 设置运行设备，优先使用CUDA，没有则使用CPU
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def cos_sim(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
    """
    计算两个向量的余弦相似度
    :return: 余弦相似度矩阵，res[i][j] = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def semantic_search(
        query_embeddings: Union[torch.Tensor, np.ndarray],
        corpus_embeddings: Union[torch.Tensor, np.ndarray],
        query_chunk_size: int = 100,
        corpus_chunk_size: int = 500000,
        top_k: int = 10,
        score_function=cos_sim
):
    """
    进行语义搜索，使用余弦相似度搜索查询嵌入和语料嵌入
    :param query_embeddings: 查询嵌入的2维张量
    :param corpus_embeddings: 语料嵌入的2维张量
    :param query_chunk_size: 同时处理的查询数量
    :param corpus_chunk_size: 一次扫描语料库的条目数量
    :param top_k: 返回最匹配的前k个条目
    :param score_function: 计算分数的函数，默认为余弦相似度
    :return: 返回按余弦相似度降序排列的结果列表，列表项为包含'corpus_id'和'score'键的字典
    """

    # 将NumPy数组转换为PyTorch张量
    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    # 如果查询嵌入是列表，则将其堆叠成张量
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    # 如果查询嵌入是一维的，则增加一个维度
    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    # 将NumPy数组转换为PyTorch张量
    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    # 如果语料嵌入是列表，则将其堆叠成张量
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # 确保查询嵌入和语料嵌入在同一设备上
    query_embeddings = query_embeddings.to(device)
    corpus_embeddings = corpus_embeddings.to(device)

    # 初始化存储查询结果的列表
    queries_result_list = [[] for _ in range(len(query_embeddings))]

    # 对查询嵌入进行分块处理
    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # 对语料嵌入进行分块处理
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # 计算余弦相似度
            cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                        corpus_embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])

            # 获取前k个最高分数
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])),
                                                                       dim=1, largest=True, sorted=False)
            # 将分数转换为列表形式
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            # 将索引转换为列表形式
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            # 遍历每个查询结果
            for query_itr in range(len(cos_scores)):
                # 遍历每个最高分数
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr],
                                                cos_scores_top_k_values[query_itr]):
                    # 计算语料库中的实际ID
                    corpus_id = corpus_start_idx + sub_corpus_id
                    # 计算查询的实际ID
                    query_id = query_start_idx + query_itr
                    # 添加结果到查询结果列表
                    queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})

    # 对每个查询结果进行排序，并只保留前k个结果
    for idx in range(len(queries_result_list)):
        queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
        queries_result_list[idx] = queries_result_list[idx][0:top_k]

    # 返回查询结果列表
    return queries_result_list



class SimilarityABC:
    """
    相似度计算和搜索的接口。

    所有实例都有一个语料库，用于执行相似度搜索。
    对于每次搜索，输入是文档或语料库，输出是与个别语料文档的相似度。
    """

    def add_corpus(self, corpus: Union[List[str], Dict[str, str]]):
        """
        将新文档扩展到语料库中。

        参数
        ----------
        corpus : str的列表或字典
        """
        raise NotImplementedError("不能实例化抽象基类")

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        计算两个文本之间的相似度。
        :param a: str的列表或单个str
        :param b: str的列表或单个str
        :param score_function: 计算相似度的函数，默认为cos_sim
        :return: 相似度得分，torch.Tensor，矩阵中的res[i][j] = cos_sim(a[i], b[j])
        """
        raise NotImplementedError("不能实例化抽象基类")

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """
        计算两个文本之间的余弦距离。
        """
        raise NotImplementedError("不能实例化抽象基类")

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """
        找出与查询最相似的topn个文本。
        :param queries: 查询的字典（查询ID: 查询文本）或字符串列表或单个字符串
        :param topn: int
        :return: 字典，格式为{查询ID: {语料库ID: 相似度得分}, ...}
        """
        raise NotImplementedError("不能实例化抽象基类")

    def search(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """
        找出与查询最相似的topn个文本。
        :param queries: 查询的字典（查询ID: 查询文本）或字符串列表或单个字符串
        :param topn: int
        :return: 字典，格式为{查询ID: {语料库ID: 相似度得分}, ...}
        """
        return self.most_similar(queries, topn=topn)



class WordEmbeddingSimilarity(SimilarityABC):
    """
    计算两个句子之间的Word2Vec相似度，并从给定语料库中检索最相似的句子。
    """

    def __init__(self, corpus: Union[List[str], Dict[str, str]] = None, model_name_or_path="w2v-light-tencent-chinese"):
        """
        初始化 WordEmbeddingSimilarity。
        :param model_name_or_path: Word2Vec模型的名称或模型文件路径。
        :param corpus: 字符串列表
        """
        try:
            from text2vec import Word2Vec
        except ImportError:
            raise ImportError("请先安装text2vec，命令为：`pip install text2vec`")
        # 加载Word2Vec模型
        if isinstance(model_name_or_path, str):
            self.keyedvectors = Word2Vec(model_name_or_path)
        elif hasattr(model_name_or_path, "encode"):
            self.keyedvectors = model_name_or_path
        else:
            raise ValueError("model_name_or_path必须是~text2vec.Word2Vec或Word2Vec模型名称")
        self.corpus = {}  # 初始化语料库

        self.corpus_embeddings = []  # 初始化语料库嵌入
        if corpus is not None:  # 如果提供了语料库，则添加到类中
            self.add_corpus(corpus)

    def __len__(self):
        """获取语料库的长度。"""
        return len(self.corpus)

    def __str__(self):
        """定义类的字符串表示。"""
        base = f"相似度: {self.__class__.__name__}, 匹配模型: Word2Vec"
        if self.corpus:
            base += f", 语料库大小: {len(self.corpus)}"
        return base

    def add_corpus(self, corpus: Union[List[str], Dict[str, str]]):
        """
        将新文档扩展到语料库中。

        参数
        ----------
        corpus : 字符串列表
        """
        corpus_new = {}
        start_id = len(self.corpus) if self.corpus else 0
        if isinstance(corpus, list):
            corpus = list(set(corpus))
            for id, doc in enumerate(corpus):
                if doc not in list(self.corpus.values()):
                    corpus_new[start_id + id] = doc
        else:
            for id, doc in corpus.items():
                if doc not in list(self.corpus.values()):
                    corpus_new[id] = doc
        self.corpus.update(corpus_new)

        logger.info(f"Start computing corpus embeddings, new docs: {len(corpus_new)}")
        corpus_texts = list(corpus_new.values())
        corpus_embeddings = self._get_vector(corpus_texts, show_progress_bar=True).tolist()
        if self.corpus_embeddings:
            self.corpus_embeddings += corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(corpus)} docs, total: {len(self.corpus)}, emb size: {len(self.corpus_embeddings)}")

    def _get_vector(self, text, show_progress_bar: bool = False) -> np.ndarray:
        """获取文本的嵌入向量。"""
        return self.keyedvectors.encode(text, show_progress_bar=show_progress_bar)

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """计算两个文本之间的余弦相似度。"""
        v1 = self._get_vector(a)
        v2 = self._get_vector(b)
        return cos_sim(v1, v2)

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """计算两个文本之间的余弦距离。"""
        return 1 - self.similarity(a, b)

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10):
        """
        找出与查询最相似的topn个文本。
        :param queries: 查询的字符串列表或单个字符串
        :param topn: int
        :return: 元组列表（语料库ID, 语料库文本, 相似度得分）
        """
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}
        queries_ids_map = {i: id for i, id in enumerate(list(queries.keys()))}
        queries_texts = list(queries.values())

        queries_embeddings = np.array([self._get_vector(query) for query in queries_texts], dtype=np.float32)
        corpus_embeddings = np.array(self.corpus_embeddings, dtype=np.float32)
        all_hits = semantic_search(queries_embeddings, corpus_embeddings, top_k=topn)
        for idx, hits in enumerate(all_hits):
            for hit in hits[0:topn]:
                result[queries_ids_map[idx]][hit['corpus_id']] = hit['score']

        return result
