import json

# 从JSON文件中读取数据
with open('/home/djh/code/RAG_news/data/news.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 初始化索引字典
title_to_ids = {}
description_to_ids = {}
body_to_ids = {}
keywords_to_ids = {}
theme_to_ids = {}
link_to_ids = {}

# 遍历数据并填充索引
for item in data:
    title_to_ids.setdefault(item['Title'], []).append(item['ID'])
    description_to_ids.setdefault(item['Description'], []).append(item['ID'])
    body_to_ids.setdefault(item['Body'], []).append(item['ID'])
    keywords_to_ids.setdefault(item['Keywords'], []).append(item['ID'])
    theme_to_ids.setdefault(item['Theme'], []).append(item['ID'])
    link_to_ids.setdefault(item['Link'], []).append(item['ID'])

# 保存每个索引到不同的JSON文件
def save_to_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

save_to_json(title_to_ids, '/home/djh/code/RAG_news/data/title_to_ids.json')
save_to_json(description_to_ids, '/home/djh/code/RAG_news/data/description_to_ids.json')
save_to_json(body_to_ids, '/home/djh/code/RAG_news/data/body_to_ids.json')
save_to_json(keywords_to_ids, '/home/djh/code/RAG_news/data/keywords_to_ids.json')
save_to_json(theme_to_ids, '/home/djh/code/RAG_news/data/theme_to_ids.json')
save_to_json(link_to_ids, '/home/djh/code/RAG_news/data/link_to_ids.json')
