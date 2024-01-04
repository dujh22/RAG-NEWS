import csv
import json

# 打开CSV文件
with open('/home/djh/code/RAG_news/data/news.csv', 'r', encoding='windows-1252') as csv_file:
    # 读取CSV文件内容
    csv_reader = csv.DictReader(csv_file)
    
    # 创建一个列表，用于存储所有行的数据
    data = []
    for row in csv_reader:
        # 将每一行的数据添加到列表中
        data.append(row)

# 将数据作为一个JSON列表写入文件
with open('/home/djh/code/RAG_news/data/news.json', 'w', encoding='utf-8') as json_file:
    json.dump(data, json_file, ensure_ascii=False, indent=4)
