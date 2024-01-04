from BaseOnly import (news, keywords_to_ids2_index,
                      theme_index, WordModel,
                      body_model, body_index,
                      description_model, description_index,
                      keywordss_model, keywordss_index,
                      title_model, title_index)
# 检索用

# 单独关键词
def rag_keyword_only(query, topn = 1):
    res = WordModel.most_similar(query, topn)
    for q_id, c in res.items():
        for corpus_id, s in c.items():
            info = keywords_to_ids2_index.get(WordModel.corpus[corpus_id])
            grade = s
            break
        break

    rag = []
    if info:
        for i in info:
            rag.append(news[int(i) - 1])
    return rag

# 单独主题词
def rag_theme_only(query, topn = 1):
    res = WordModel.most_similar(query, topn)
    for q_id, c in res.items():
        for corpus_id, s in c.items():
            info = theme_index.get(WordModel.corpus[corpus_id])
            grade = s
            break
        break

    rag = []
    if info:
        for i in info:
            rag.append(news[int(i) - 1])
    return rag

# 正文
def rag_body(query, topn = 1):
    res = body_model.most_similar(query, topn)

    rag = []

    for q_id, c in res.items():
        for corpus_id, s in c.items():
            rag.append(news[int(body_index[body_model.corpus[corpus_id]][0]) - 1])
    return rag

def rag_description(query, topn = 1):
    res = description_model.most_similar(query, topn)

    rag = []

    for q_id, c in res.items():
        for corpus_id, s in c.items():
            rag.append(news[int(description_index[description_model.corpus[corpus_id]][0]) - 1])
    return rag

def rag_keywordss(query, topn = 1):
    res = keywordss_model.most_similar(query, topn)

    rag = []

    for q_id, c in res.items():
        for corpus_id, s in c.items():
            rag.append(news[int(keywordss_index[keywordss_model.corpus[corpus_id]][0]) - 1])
    return rag

def rag_title(query, topn = 1):
    res = title_model.most_similar(query, topn)

    rag = []

    for q_id, c in res.items():
        for corpus_id, s in c.items():
            rag.append(news[int(title_index[title_model.corpus[corpus_id]][0]) - 1])
    return rag

if __name__ == "__main__":
    query = ("115 improperly stored human remains found in Colorado funeral home, sheriff says ")
    for i in rag_title(query, 3):
        print(i["Title"])
