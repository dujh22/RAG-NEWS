import json

# 读取已有的keywords_to_ids.json文件
with open('/home/djh/code/RAG_news/data/keywords_to_ids.json', 'r', encoding='utf-8') as file:
    keywords_to_ids = json.load(file)

# 初始化一个新的字典来存储处理后的关键词索引
processed_keywords_to_ids = {}

# 分割关键词并更新索引
for keywords, ids in keywords_to_ids.items():
    # 分割关键词字符串
    individual_keywords = keywords.split(', ')
    for keyword in individual_keywords:
        # 如果关键词已存在，则追加ID；否则，创建新的键值对
        if keyword in processed_keywords_to_ids:
            processed_keywords_to_ids[keyword].extend(ids)
        else:
            processed_keywords_to_ids[keyword] = ids

# 移除重复的ID
for keyword, ids in processed_keywords_to_ids.items():
    processed_keywords_to_ids[keyword] = list(set(ids))

# 将处理后的数据保存回JSON文件
with open('/home/djh/code/RAG_news/data/keywords_to_ids2.json', 'w', encoding='utf-8') as file:
    json.dump(processed_keywords_to_ids, file, ensure_ascii=False, indent=4)
