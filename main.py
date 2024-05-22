# -*- coding: utf-8 -*-

# 导入模块
import json
import re
import tiktoken
from tqdm import tqdm
from openai_summarize import openai_summarize


def load_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        txtdata = f.read()
    return txtdata


def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# 小说拆分区块
def get_chunk(text, enc=tiktoken.get_encoding("cl100k_base")):
    max_token_len = 600
    chunk_text = []

    curr_len = 0
    curr_chunk = ''

    lines = text.split('\n')  # 假设以换行符分割文本为行

    for line in lines:
        # 跳过带有“第 章”的行
        if re.search(r"第.*?章", line):
            continue

        line_len = len(enc.encode(line))
        if line_len > max_token_len:
            print('warning line_len = ', line_len)
        if curr_len + line_len <= max_token_len:
            curr_chunk += line
            curr_chunk += '\n'
            curr_len += line_len
            curr_len += 1
        else:
            chunk_text.append(curr_chunk)
            curr_chunk = line
            curr_len = line_len

    if curr_chunk:
        chunk_text.append(curr_chunk)

    return chunk_text


# 保存为jsonl格式
def save_to_jsonl(chunks, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            json_line = {"id": str(i), "text": chunk}
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')



# 入口

if __name__ == '__main__':
    # 将文本分段
    bookname = "book"
    text = load_txt(f'input/{bookname}.txt')
    chunk_text = get_chunk(text)
    save_to_jsonl(chunk_text, f'output/{bookname}.jsonl')
    print("======================================文本分段成功======================================")
    # 正文概括
    llm = openai_summarize()
    chunk_jsonl = load_jsonl(f'output/{bookname}.jsonl')

    with open(f'output/{bookname}_s.jsonl', 'a', encoding='utf-8') as file:
        for i in tqdm(range(len(chunk_jsonl))):
            MAX_COUNT = 3
            current_count = 1
            response = ""
            while current_count <= MAX_COUNT:
                try:
                    text_value = chunk_jsonl[i].get("text")
                    prompt = f"概括：{text_value}"
                    response = llm._call(prompt)
                    # print(response)
                except Exception as e:
                    print(e)
                else:
                    break
                finally:
                    print(f"第 {current_count} 次尝试完成。")
                    current_count += 1
            if current_count > MAX_COUNT:
                message = " "
                json_line = {"id": str(i), "summarize": message}
                file.write(json.dumps(json_line, ensure_ascii=False) + '\n')
                continue
            message = response.choices[0].message.content
            json_line = {"id": str(i), "summarize": message}
            file.write(json.dumps(json_line, ensure_ascii=False) + '\n')




