#!/usr/bin/env python3
# coding: utf-8

# 本脚本适用于pytho3.10及以上
# requests和json是本脚本的依赖，需要安装
import requests
import json
import streamlit as st
from search import *

st.set_page_config(page_title="RAG NEWS")
st.title("RAG NEWS")

url = 'http://192.168.0.118:8055/v1/chat/completions'

header = {
    'Content-Type': 'application/json'
}
data = {
    'model': "Baichuan",
    'stream': True,
    'temperature': 1
}

# 清空对话
def clear_chat_history():
    del st.session_state.messages

# 初始化历史信息显示
def init_chat_history():
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    competiton_types = ['查询与新闻检索无关', '单一关键词检索', '单一主题词检索',
                        '新闻标题检索', '新闻描述检索', '多关键词的检索', '新闻正文检索']
    competition_type = st.selectbox('请选择具体需求', competiton_types)

    if competition_type == '查询与新闻检索无关':
        # 具体交互过程
        with st.chat_message("assistant", avatar='🤖'):
            st.markdown("你好, 这里是RAG NESW. 什么新闻是你今天关注的呢?")
        messages = init_chat_history()

        if prompt := st.chat_input("Shift + Enter to line feed, Enter to send"):
            with st.chat_message("user", avatar='🧑‍💻'):
                st.markdown(prompt)
            messages.append({"role": "user", "content": prompt})

            with st.chat_message("assistant", avatar='🤖'):
                placeholder = st.empty()

                data["messages"] = messages
                response = requests.post(url, headers=header, json=data, stream=True)
                answer = ""
                for chunk in response.iter_content(chunk_size=1024):  # 注意chunk是bytes类型，需要先转换
                    # 处理响应内容
                    # 将bytes类型转换为字符串类型
                    str_obj = chunk.decode('utf-8')
                    str_obj = str_obj.replace('data: ', '').strip()
                    # 将字符串类型转换为json格式
                    json_obj = json.loads(str_obj)

                    if 'content' in json_obj['choices'][0]['delta'].keys():
                        answer = answer + json_obj['choices'][0]['delta']['content']
                        placeholder.markdown(answer)
            messages.append({"role": "assistant", "content": answer})

        st.button("清空对话", on_click=clear_chat_history)
    elif competition_type == '单一关键词检索':
        topk = st.number_input('检索召回数量', min_value=1, max_value=5, value=1)

        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            placeholder.markdown("请输入单个关键词")

        # 具体交互过程
        messages = init_chat_history()

        if prompt := st.chat_input("Shift + Enter to line feed, Enter to send"):
            with st.chat_message("user", avatar='🧑‍💻'):
                st.markdown(prompt)
            rag_prompt = str(rag_keyword_only(prompt, topk)) + "请进行总结综述"
            messages.append({"role": "user", "content": rag_prompt})

            with st.chat_message("assistant", avatar='🤖'):
                placeholder = st.empty()

                data["messages"] = messages
                response = requests.post(url, headers=header, json=data, stream=True)
                answer = ""
                for chunk in response.iter_content(chunk_size=1024):  # 注意chunk是bytes类型，需要先转换
                    # 处理响应内容
                    # 将bytes类型转换为字符串类型
                    str_obj = chunk.decode('utf-8')
                    str_obj = str_obj.replace('data: ', '').strip()
                    # 将字符串类型转换为json格式
                    json_obj = json.loads(str_obj)

                    if 'content' in json_obj['choices'][0]['delta'].keys():
                        answer = answer + json_obj['choices'][0]['delta']['content']
                        placeholder.markdown(answer)
            messages.append({"role": "assistant", "content": answer})

    elif competition_type == '单一主题词检索':
        topk = st.number_input('检索召回数量', min_value=1, max_value=5, value=1)

        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            placeholder.markdown("请输入单个主题词")

        # 具体交互过程
        messages = init_chat_history()

        if prompt := st.chat_input("Shift + Enter to line feed, Enter to send"):
            with st.chat_message("user", avatar='🧑‍💻'):
                st.markdown(prompt)
            rag_prompt = str(rag_theme_only(prompt, topk))+ "请进行总结综述"
            messages.append({"role": "user", "content": rag_prompt})

            with st.chat_message("assistant", avatar='🤖'):
                placeholder = st.empty()

                data["messages"] = messages
                response = requests.post(url, headers=header, json=data, stream=True)
                answer = ""
                for chunk in response.iter_content(chunk_size=1024):  # 注意chunk是bytes类型，需要先转换
                    # 处理响应内容
                    # 将bytes类型转换为字符串类型
                    str_obj = chunk.decode('utf-8')
                    str_obj = str_obj.replace('data: ', '').strip()
                    # 将字符串类型转换为json格式
                    json_obj = json.loads(str_obj)

                    if 'content' in json_obj['choices'][0]['delta'].keys():
                        answer = answer + json_obj['choices'][0]['delta']['content']
                        placeholder.markdown(answer)
            messages.append({"role": "assistant", "content": answer})

    elif competition_type == '新闻标题检索':
        topk = st.number_input('检索召回数量', min_value=1, max_value=5, value=1)

        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            placeholder.markdown("请输入新闻标题")

        # 具体交互过程
        messages = init_chat_history()

        if prompt := st.chat_input("Shift + Enter to line feed, Enter to send"):
            with st.chat_message("user", avatar='🧑‍💻'):
                st.markdown(prompt)
            rag_prompt = str(rag_title(prompt, topk))+ "请进行总结综述"
            messages.append({"role": "user", "content": rag_prompt})

            with st.chat_message("assistant", avatar='🤖'):
                placeholder = st.empty()

                data["messages"] = messages
                response = requests.post(url, headers=header, json=data, stream=True)
                answer = ""
                for chunk in response.iter_content(chunk_size=1024):  # 注意chunk是bytes类型，需要先转换
                    # 处理响应内容
                    # 将bytes类型转换为字符串类型
                    str_obj = chunk.decode('utf-8')
                    str_obj = str_obj.replace('data: ', '').strip()
                    # 将字符串类型转换为json格式
                    json_obj = json.loads(str_obj)

                    if 'content' in json_obj['choices'][0]['delta'].keys():
                        answer = answer + json_obj['choices'][0]['delta']['content']
                        placeholder.markdown(answer)
            messages.append({"role": "assistant", "content": answer})

    elif competition_type == '新闻描述检索':
        topk = st.number_input('检索召回数量', min_value=1, max_value=5, value=1)

        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            placeholder.markdown("请输入新闻描述")

        # 具体交互过程
        messages = init_chat_history()

        if prompt := st.chat_input("Shift + Enter to line feed, Enter to send"):
            with st.chat_message("user", avatar='🧑‍💻'):
                st.markdown(prompt)
            rag_prompt = str(rag_description(prompt, topk))+ "请进行总结综述"
            messages.append({"role": "user", "content": rag_prompt})

            with st.chat_message("assistant", avatar='🤖'):
                placeholder = st.empty()

                data["messages"] = messages
                response = requests.post(url, headers=header, json=data, stream=True)
                answer = ""
                for chunk in response.iter_content(chunk_size=1024):  # 注意chunk是bytes类型，需要先转换
                    # 处理响应内容
                    # 将bytes类型转换为字符串类型
                    str_obj = chunk.decode('utf-8')
                    str_obj = str_obj.replace('data: ', '').strip()
                    # 将字符串类型转换为json格式
                    json_obj = json.loads(str_obj)

                    if 'content' in json_obj['choices'][0]['delta'].keys():
                        answer = answer + json_obj['choices'][0]['delta']['content']
                        placeholder.markdown(answer)
            messages.append({"role": "assistant", "content": answer})

    elif competition_type == '多关键词的检索':
        topk = st.number_input('检索召回数量', min_value=1, max_value=5, value=1)

        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            placeholder.markdown("请一次性输入多个关键词")

        # 具体交互过程
        messages = init_chat_history()

        if prompt := st.chat_input("Shift + Enter to line feed, Enter to send"):
            with st.chat_message("user", avatar='🧑‍💻'):
                st.markdown(prompt)
            rag_prompt = str(rag_keywordss(prompt, topk))+ "请进行总结综述"
            messages.append({"role": "user", "content": rag_prompt})

            with st.chat_message("assistant", avatar='🤖'):
                placeholder = st.empty()

                data["messages"] = messages
                response = requests.post(url, headers=header, json=data, stream=True)
                answer = ""
                for chunk in response.iter_content(chunk_size=1024):  # 注意chunk是bytes类型，需要先转换
                    # 处理响应内容
                    # 将bytes类型转换为字符串类型
                    str_obj = chunk.decode('utf-8')
                    str_obj = str_obj.replace('data: ', '').strip()
                    # 将字符串类型转换为json格式
                    json_obj = json.loads(str_obj)

                    if 'content' in json_obj['choices'][0]['delta'].keys():
                        answer = answer + json_obj['choices'][0]['delta']['content']
                        placeholder.markdown(answer)
            messages.append({"role": "assistant", "content": answer})

    elif competition_type == '新闻正文检索':
        topk = st.number_input('检索召回数量', min_value=1, max_value=5, value=1)

        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            placeholder.markdown("请输入新闻正文")

        # 具体交互过程
        messages = init_chat_history()

        if prompt := st.chat_input("Shift + Enter to line feed, Enter to send"):
            with st.chat_message("user", avatar='🧑‍💻'):
                st.markdown(prompt)
            rag_prompt = str(rag_body(prompt, topk))+ "请进行总结综述"
            messages.append({"role": "user", "content": rag_prompt})

            with st.chat_message("assistant", avatar='🤖'):
                placeholder = st.empty()

                data["messages"] = messages
                response = requests.post(url, headers=header, json=data, stream=True)
                answer = ""
                for chunk in response.iter_content(chunk_size=1024):  # 注意chunk是bytes类型，需要先转换
                    # 处理响应内容
                    # 将bytes类型转换为字符串类型
                    str_obj = chunk.decode('utf-8')
                    str_obj = str_obj.replace('data: ', '').strip()
                    # 将字符串类型转换为json格式
                    json_obj = json.loads(str_obj)

                    if 'content' in json_obj['choices'][0]['delta'].keys():
                        answer = answer + json_obj['choices'][0]['delta']['content']
                        placeholder.markdown(answer)
            messages.append({"role": "assistant", "content": answer})



if __name__ == "__main__":
    main()
