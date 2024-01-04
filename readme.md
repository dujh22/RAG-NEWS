# RAG_NEWS

本项目实现了一个简单的 RAG 框架，并使 LLM 能够获取有关 2023 年新闻的信息。

## 项目环境配置

1. 请先到[基座模型百川2网址](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat/tree/main)下载全部模型bin文件到Baichuan-13B-Chat文件夹内，
同时到[shibing624/text2vec-base-chinese-paraphrase](https://huggingface.co/shibing624/text2vec-base-chinese-paraphrase)下载文本嵌入模型到text2vec-base-chinese-paraphrase文件夹内

2. python选用3.11.5(3.11均可)

3. 安装pytorch

假设本地机器CUDA版本最高支持为11.7, 我们希望尽可能安装可支持的最新的pytorch版本，比如2.0.1，具体下载命令参照：https://pytorch.org/get-started/previous-versions/

```shell
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

4. 推理前请安装依赖：
```shell
pip install -r requirements.txt
```

## 项目构成说明
| 名称                                                         | 用途                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Baichuan-13B-Chat文件夹，text2vec-base-chinese-paraphrase文件夹 | 包括项目使用的各种模型的参数数据                             |
| data文件夹                                                   | 包括项目使用的原始数据和处理后的数据，其中news.csv是原始数据 |
| data_to_json.py                                              | 用于将原始数据转换为json格式                                 |
| json_split.py                                                | 用于将原始数据的json文件根据不同键为索引重新拆分成几个独立的json，方便后续检索和训练 |
| keywords_split.py                                            | 考虑某些键为多个字符串的合并，比如关键词串，需要进一步拆分重组 |
| openai_api.py                                                | 按照openai标准实现的baichuan流式访问接口，需要最先独立启动：python openai_api.py |
| word_similarity.py                                           | 针对词类型的检索实现的相关算法                               |
| sentence_similarit.py                                        | 针对句子或者篇章类型的检索实现的相关算法                     |
| BaseOnly.py                                                  | 统一对数据的读取、对模型的读取                               |
| search.py                                                    | 大模型调用所有检索算法的中间件                               |
| front.py                                                     | 前端，启动后相应后端也会同步启动，具体启动方式： streamlit run front.py --server.port 8008 |

