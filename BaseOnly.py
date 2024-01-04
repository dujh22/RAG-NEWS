from sentence_similarity import AnnoySimilarity
import json

# 加载数据库
news_place = '/home/djh/code/RAG_news/data/news.json'
with open(news_place, 'r', encoding='utf-8') as news_place:
    news = json.load(news_place)

# 词模型
from word_similarity import WordEmbeddingSimilarity
WordModel = WordEmbeddingSimilarity(model_name_or_path = '/home/djh/code/RAG_news/text2vec-base-chinese-paraphrase/pytorch_model')

# 词检索
# 单独关键词
keywords_to_ids2_index_place = '/home/djh/code/RAG_news/data/keywords_to_ids2.json'
with open(keywords_to_ids2_index_place, 'r', encoding='utf-8') as keywords_to_ids2_index_f:
    keywords_to_ids2_index = json.load(keywords_to_ids2_index_f)
WordModel.add_corpus(list(keywords_to_ids2_index.keys()))
# 单独主题词
theme_index_place = '/home/djh/code/RAG_news/data/theme_to_ids.json'
with open(theme_index_place, 'r', encoding='utf-8') as theme_index_f:
    theme_index = json.load(theme_index_f)
WordModel.add_corpus(list(theme_index.keys()))

# 句检索
# 正文
body_index_place = "/home/djh/code/RAG_news/data/body_to_ids.json"
with open(body_index_place, 'r', encoding='utf-8') as sentence_index_body:
    body_index = json.load(sentence_index_body)
body_corpus = list(body_index.keys())
body_model = AnnoySimilarity(corpus=body_corpus)
body_model.build_index()
body_index_file = '/home/djh/code/RAG_news/data/body_annoy_model.index'
body_model.save_index(body_index_file)
# 描述
description_index_place = "/home/djh/code/RAG_news/data/description_to_ids.json"
with open(description_index_place, 'r', encoding='utf-8') as sentence_index_description:
    description_index = json.load(sentence_index_description)
description_corpus = list(description_index.keys())
description_model = AnnoySimilarity(corpus=description_corpus)
description_model.build_index()
description_index_file = '/home/djh/code/RAG_news/data/description_annoy_model.index'
description_model.save_index(description_index_file)
# 多关键词
keywordss_index_place = "/home/djh/code/RAG_news/data/keywords_to_ids.json"
with open(keywordss_index_place, 'r', encoding='utf-8') as sentence_index_keywordss:
    keywordss_index = json.load(sentence_index_keywordss)
keywordss_corpus = list(keywordss_index.keys())
keywordss_model = AnnoySimilarity(corpus=keywordss_corpus)
keywordss_model.build_index()
keywordss_index_file = '/home/djh/code/RAG_news/data/keywordss_annoy_model.index'
keywordss_model.save_index(keywordss_index_file)
# 标题
title_index_place = "/home/djh/code/RAG_news/data/title_to_ids.json"
with open(title_index_place, 'r', encoding='utf-8') as sentence_index_title:
    title_index = json.load(sentence_index_title)
title_corpus = list(title_index.keys())
title_model = AnnoySimilarity(corpus=title_corpus)
title_model.build_index()
title_index_file = '/home/djh/code/RAG_news/data/title_annoy_model.index'
title_model.save_index(title_index_file)

