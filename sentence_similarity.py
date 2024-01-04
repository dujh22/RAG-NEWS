# -*- coding: utf-8 -*-

import json
import os
import sys
from loguru import logger
from typing import List, Union, Dict
import numpy as np
from text2vec import SentenceModel

import queue

import numpy as np
import torch
import torch.nn.functional

# 设置计算设备，优先使用CUDA GPU，如果没有则使用CPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# 自定义模型路径
model_path_self = "/home/djh/code/RAG_news/text2vec-base-chinese-paraphrase"


# 将上级目录加入到系统路径中，以便导入上级目录的模块
sys.path.append('..')

# 设置环境变量，用于处理一些库的兼容性问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "TRUE"

def cos_sim(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
    """
    计算两个向量间的余弦相似度。
    :return: 相似度矩阵，res[i][j] = cos_sim(a[i], b[j])
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

def dot_score(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):
    """
    计算两个向量间的点积。
    :return: 点积矩阵，res[i][j] = dot_prod(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))

def semantic_search(
        query_embeddings: Union[torch.Tensor, np.ndarray],
        corpus_embeddings: Union[torch.Tensor, np.ndarray],
        query_chunk_size: int = 100,
        corpus_chunk_size: int = 500000,
        top_k: int = 10,
        score_function=cos_sim
):
    """
    执行语义搜索，通过余弦相似度在查询嵌入和语料嵌入间进行搜索。
    :param query_embeddings: 查询嵌入的二维张量。
    :param corpus_embeddings: 语料嵌入的二维张量。
    :param query_chunk_size: 同时处理的查询数量。
    :param corpus_chunk_size: 一次处理的语料库条目数量。
    :param top_k: 返回匹配的前k个条目。
    :param score_function: 计算分数的函数，默认为余弦相似度。
    :return: 按相似度降序排列的结果列表，列表项为包含'corpus_id'和'score'键的字典。
    """

    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)

    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # 检查语料库和查询是否在同一设备上
    query_embeddings = query_embeddings.to(device)
    corpus_embeddings = corpus_embeddings.to(device)

    queries_result_list = [[] for _ in range(len(query_embeddings))]

    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        # 迭代语料块
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            # 计算余弦相似度
            cos_scores = score_function(query_embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                        corpus_embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])

            # 获取最高 k 分数
            cos_scores_top_k_values, cos_scores_top_k_idx = torch.topk(cos_scores, min(top_k, len(cos_scores[0])),
                                                                       dim=1, largest=True, sorted=False)
            cos_scores_top_k_values = cos_scores_top_k_values.cpu().tolist()
            cos_scores_top_k_idx = cos_scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(cos_scores)):
                for sub_corpus_id, score in zip(cos_scores_top_k_idx[query_itr],
                                                cos_scores_top_k_values[query_itr]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_itr
                    queries_result_list[query_id].append({'corpus_id': corpus_id, 'score': score})

    # 排序并剥离到结果的前千位
    for idx in range(len(queries_result_list)):
        queries_result_list[idx] = sorted(queries_result_list[idx], key=lambda x: x['score'], reverse=True)
        queries_result_list[idx] = queries_result_list[idx][0:top_k]

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

class BertSimilarity(SimilarityABC):
    """
    句子相似度计算：
    1. 计算两个句子之间的相似度。
    2. 在文档语料库中检索查询句子的最相似句子。

    支持动态添加新文档到索引。
    """

    def __init__(
            self,
            corpus: Union[List[str], Dict[str, str]] = None,
            model_name_or_path=model_path_self,
            device=None,
    ):
        """
        初始化相似性对象。
        :param model_name_or_path： 转换器模型名称或路径，如
            sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'bert-base-uncased', 'bert-base-chinese'、
             shibing624/text2vec-base-chinese', ...
            在 HuggingFace Model Hub 中建立模型，并从 https://github.com/shibing624/text2vec 发布
        :param corpus： 用于相似性查询的文档语料库。
        :param device：设备： 用于计算的设备（如 'cuda' / 'cpu'）。
        """
        if isinstance(model_name_or_path, str):
            self.sentence_model = SentenceModel(
                model_name_or_path,
                device=device
            )
        elif hasattr(model_name_or_path, "encode"):
            self.sentence_model = model_name_or_path
        else:
            raise ValueError("model_name_or_path 是transformers的模型名称或路径")
        self.score_functions = {'cos_sim': cos_sim, 'dot': dot_score}
        self.corpus = {}
        self.corpus_embeddings = []
        if corpus is not None:
            self.add_corpus(corpus)

    def __len__(self):
        """获取语料长度。"""
        return len(self.corpus)

    def __str__(self):
        base = f"Similarity: {self.__class__.__name__}, matching_model: {self.sentence_model}"
        if self.corpus:
            base += f", corpus size: {len(self.corpus)}"
        return base

    def get_sentence_embedding_dimension(self):
        """
        获取句子嵌入维度。

        返回
        -------
        整数或无
            句子嵌入的维度，如果无法确定，则返回 None。
        """
        if hasattr(self.sentence_model, "获取句子嵌入维度"):
            return self.sentence_model.get_sentence_embedding_dimension()
        else:
            return getattr(self.sentence_model.bert.pooler.dense, "out_features", None)

    def add_corpus(self, corpus: Union[List[str], Dict[str, str]]):
        """
        使用新文档扩展语料库。
        :param corpus：用于相似性查询的语料库。
        :return: self.corpus, self.corpus embeddings
        """
        new_corpus = {}
        start_id = len(self.corpus) if self.corpus else 0
        for id, doc in enumerate(corpus):
            if isinstance(corpus, list):
                if doc not in self.corpus.values():
                    new_corpus[start_id + id] = doc
            else:
                if doc not in self.corpus.values():
                    new_corpus[id] = doc
        self.corpus.update(new_corpus)
        logger.info(f"Start computing corpus embeddings, new docs: {len(new_corpus)}")
        corpus_embeddings = self.get_embeddings(list(new_corpus.values()), show_progress_bar=True).tolist()
        if self.corpus_embeddings:
            self.corpus_embeddings = self.corpus_embeddings + corpus_embeddings
        else:
            self.corpus_embeddings = corpus_embeddings
        logger.info(f"Add {len(new_corpus)} docs, total: {len(self.corpus)}, emb len: {len(self.corpus_embeddings)}")

    def get_embeddings(
            self,
            sentences: Union[str, List[str]],
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = False,
    ):
        """
        返回一批句子的嵌入结果。
        """
        return self.sentence_model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            convert_to_tensor=convert_to_tensor,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )

    def similarity(self, a: Union[str, List[str]], b: Union[str, List[str]], score_function: str = "cos_sim"):
        """
        计算两个文本的相似性。
        :param a: list of str or str
        :param b：字符串或字符串列表
        :param score_function：计算相似度的函数，默认为 cos_sim
        返回：相似度得分，torrent.Tensor，矩阵，res[i][j] = cos_sim(a[i], b[j])
        """
        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity"
                             " or (dot) for dot product")
        score_function = self.score_functions[score_function]
        text_emb1 = self.get_embeddings(a)
        text_emb2 = self.get_embeddings(b)

        return score_function(text_emb1, text_emb2)

    def distance(self, a: Union[str, List[str]], b: Union[str, List[str]]):
        """Compute cosine distance between two texts."""
        return 1 - self.similarity(a, b)

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10,
                     score_function: str = "cos_sim"):
        """
        根据语料库查找与查询最相似的文本。
            它可用于信息检索/语义搜索，适用于多达约 100 万条目的语料库。
        :param queries: 字符串或字符串列表
        :param topn: int
        :param score_function：计算相似度的函数，默认为 cos_sim
        :return: 返回值： Dict[str, Dict[str, float]], {query_id： {corpus_id: similarity_score}, ...}
        """
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        if score_function not in self.score_functions:
            raise ValueError(f"score function: {score_function} must be either (cos_sim) for cosine similarity"
                             " or (dot) for dot product")
        score_function = self.score_functions[score_function]
        result = {qid: {} for qid, query in queries.items()}
        queries_ids_map = {i: id for i, id in enumerate(list(queries.keys()))}
        queries_texts = list(queries.values())
        queries_embeddings = self.get_embeddings(queries_texts, convert_to_tensor=True)
        corpus_embeddings = np.array(self.corpus_embeddings, dtype=np.float32)
        all_hits = semantic_search(queries_embeddings, corpus_embeddings, top_k=topn, score_function=score_function)
        for idx, hits in enumerate(all_hits):
            for hit in hits[0:topn]:
                result[queries_ids_map[idx]][hit['corpus_id']] = hit['score']

        return result

    def save_embeddings(self, emb_path: str = "corpus_emb.json"):
        """
        将语料嵌入保存到 json 文件。
        参数 emb_path：json 文件路径
        :return：
        """
        corpus_emb = {id: {"doc": self.corpus[id], "doc_emb": emb} for id, emb in
                      zip(self.corpus.keys(), self.corpus_embeddings)}
        with open(emb_path, "w", encoding="utf-8") as f:
            json.dump(corpus_emb, f, ensure_ascii=False)
        logger.debug(f"Save corpus embeddings to file: {emb_path}.")

    def load_embeddings(self, emb_path: str = "corpus_emb.json"):
        """
        从 json 文件加载语料库嵌入。
        参数 emb_path：json 文件路径
        返回：语料库嵌入列表、语料库 ID 索引图、语料库索引图。
        """
        try:
            with open(emb_path, "r", encoding="utf-8") as f:
                corpus_emb = json.load(f)
            corpus_embeddings = []
            for id, corpus_dict in corpus_emb.items():
                self.corpus[int(id)] = corpus_dict["doc"]
                corpus_embeddings.append(corpus_dict["doc_emb"])
            self.corpus_embeddings = corpus_embeddings
        except (IOError, json.JSONDecodeError):
            logger.error("Error: Could not load corpus embeddings from file.")


class AnnoySimilarity(BertSimilarity):
    """
    使用Annoy计算词嵌入之间的余弦相似度，并检索给定文档的最相似查询。
    """

    def __init__(
            self,
            corpus: Union[List[str], Dict[str, str]] = None,
            model_name_or_path=model_path_self,
            n_trees: int = 256
    ):
        """
        初始化AnnoySimilarity类。
        :param corpus: 可选，文档语料库。
        :param model_name_or_path: 模型名称或路径。
        :param n_trees: Annoy索引中的树的数量。
        """
        super().__init__(corpus, model_name_or_path)  # 调用父类构造函数
        self.index = None  # 初始化索引为空
        self.embedding_size = self.get_sentence_embedding_dimension()  # 获取句子嵌入的维度
        self.n_trees = n_trees  # 设置Annoy索引的树的数量
        if corpus is not None and self.corpus_embeddings:
            self.build_index()  # 如果提供了语料库并且有嵌入，则构建索引

    def __str__(self):
        """
        定义类的字符串表示。
        """
        base = f"相似度: {self.__class__.__name__}, 匹配模型: {self.sentence_model}"
        if self.corpus:
            base += f", 语料库大小: {len(self.corpus)}"
        return base

    def create_index(self):
        """
        创建Annoy索引。
        """
        try:
            from annoy import AnnoyIndex
        except ImportError:
            raise ImportError("Annoy未安装，请先安装，例如：`pip install annoy`。")

        # 创建Annoy索引
        self.index = AnnoyIndex(self.embedding_size, 'angular')
        logger.debug(f"初始化Annoy索引，嵌入维度: {self.embedding_size}")

    def build_index(self):
        """
        添加新文档后构建Annoy索引。
        """
        self.create_index()  # 创建索引
        logger.debug(f"使用{self.n_trees}棵树构建索引。")

        # 向索引中添加语料嵌入
        for i in range(len(self.corpus_embeddings)):
            self.index.add_item(i, self.corpus_embeddings[i])
        self.index.build(self.n_trees)  # 构建索引

    def save_index(self, index_path: str = "annoy_index.bin"):
        """
        将Annoy索引保存到磁盘。
        :param index_path: 索引文件路径。
        """
        if index_path:
            if self.index is None:
                self.build_index()  # 如果索引不存在，则构建
            self.index.save(index_path)  # 保存索引
            corpus_emb_json_path = index_path + ".json"
            super().save_embeddings(corpus_emb_json_path)  # 保存嵌入
            logger.info(f"将Annoy索引保存到: {index_path}, 语料嵌入保存到: {corpus_emb_json_path}")
        else:
            logger.warning("未给出索引路径，索引未保存。")

    def load_index(self, index_path: str = "annoy_index.bin"):
        """
        从磁盘加载Annoy索引。
        :param index_path: 索引文件路径。
        """
        if index_path and os.path.exists(index_path):
            corpus_emb_json_path = index_path + ".json"
            logger.info(f"从{index_path}加载索引，从{corpus_emb_json_path}加载语料嵌入")
            super().load_embeddings(corpus_emb_json_path)  # 加载嵌入
            if self.index is None:
                self.create_index()  # 如果索引不存在，则创建
            self.index.load(index_path)  # 加载索引
        else:
            logger.warning("未给出索引路径，索引未加载。")

    def most_similar(self, queries: Union[str, List[str], Dict[str, str]], topn: int = 10,
                     score_function: str = "cos_sim"):
        """
        找出与查询最相似的topn个文本。
        :param queries: 查询的字符串或字符串列表。
        :param topn: 返回的相似文本数量。
        :param score_function: 计算相似度的函数，默认为余弦相似度。
        :return: 相似度结果，格式为{查询ID: {语料库ID: 相似度分数}, ...}。
        """
        result = {}
        # 检查是否存在语料库嵌入和索引，如果没有索引则发出警告并使用慢速搜索
        if self.corpus_embeddings and self.index is None:
            logger.warning(f"未找到索引。请先添加语料库并构建索引，例如使用 `build_index()`。现在返回慢速搜索结果。")
            return super().most_similar(queries, topn, score_function=score_function)
        
        # 如果没有语料库嵌入，则发出错误并返回空结果
        if not self.corpus_embeddings:
            logger.error("未找到语料库嵌入。请先添加语料库，例如使用 `add_corpus()`。")
            return result

        # 处理查询数据，确保其为字典格式
        if isinstance(queries, str) or not hasattr(queries, '__len__'):
            queries = [queries]
        if isinstance(queries, list):
            queries = {id: query for id, query in enumerate(queries)}
        result = {qid: {} for qid, query in queries.items()}
        queries_texts = list(queries.values())

        # 获取查询的嵌入
        queries_embeddings = self.get_embeddings(queries_texts)

        # Annoy的get_nns_by_vector方法一次只能搜索一个向量
        for idx, (qid, query) in enumerate(queries.items()):
            # 使用Annoy索引进行搜索，返回最接近的语料库ID和它们的距离
            corpus_ids, distances = self.index.get_nns_by_vector(queries_embeddings[idx], topn, include_distances=True)
            # 将距离转换为相似度得分
            for corpus_id, distance in zip(corpus_ids, distances):
                score = 1 - (distance ** 2) / 2
                result[qid][corpus_id] = score

        return result


def annoy_demo():
    """
    Annoy演示函数。
    """
    sentences = ['如何更换花呗绑定银行卡',
                 '花呗更改绑定银行卡']
    corpus = [
        '花呗更改绑定银行卡',
        '我什么时候开通了花呗',
        '俄罗斯警告乌克兰反对欧盟协议',
        '暴风雨掩埋了东北部；新泽西16英寸的降雪',
        '中央情报局局长访问以色列叙利亚会谈',
        '人在巴基斯坦基地的炸弹袭击中丧生',
        '我喜欢这首歌'
    ]

    corpus_new = [i + str(id) for id, i in enumerate(corpus * 10)]
    print(corpus_new) # test
    model = AnnoySimilarity(corpus=corpus_new)
    print(model)
    similarity_score = model.similarity(sentences[0], sentences[1])
    print(f"{sentences[0]} vs {sentences[1]}, score: {float(similarity_score):.4f}")
    model.add_corpus(corpus)
    model.build_index()
    index_file = 'annoy_model.index'
    model.save_index(index_file)
    print(model.most_similar("men喜欢这首歌"))
    # Semantic Search batch
    del model
    model = AnnoySimilarity()
    model.load_index(index_file)
    print(model.most_similar("men喜欢这首歌"))
    queries = ["如何更换花呗绑定银行卡", "men喜欢这首歌"]
    res = model.most_similar(queries, topn=3)
    print(res)
    for q_id, c in res.items():
        print('query:', queries[q_id])
        print("search top 3:")
        for corpus_id, s in c.items():
            print(f'\t{model.corpus[corpus_id]}: {s:.4f}')

    os.remove(index_file)


if __name__ == '__main__':
    annoy_demo()
