# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
modified from https://github.com/imClumsyPanda/langchain-ChatGLM/blob/master/webui.py
"""
import argparse
import hashlib
import os
import shutil

import gradio as gr
from loguru import logger

from chatpdf import ChatPDF

pwd_path = os.path.abspath(os.path.dirname(__file__))

CONTENT_DIR = os.path.join(pwd_path, "content")

embedding_model_dict = {
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec-multilingual": "shibing624/text2vec-base-multilingual",
    "text2vec-large": "GanymedeNil/text2vec-large-chinese",
    "sentence-transformers": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
}

# supported LLM models
llm_model_dict = {
    "llama-2-7b": "LinkSoul/Chinese-Llama-2-7b-4bit",
    "llama-7b": "shibing624/chinese-alpaca-plus-7b-hf",
    "baichuan-13b-chat": "baichuan-inc/Baichuan-13B-Chat",
    "chatglm-6b": "THUDM/chatglm-6b",
}

llm_model_dict_list = list(llm_model_dict.keys())
embedding_model_dict_list = list(embedding_model_dict.keys())

block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 🎉ChatPDF WebUI🎉
Link in: [https://github.com/shibing624/ChatPDF](https://github.com/shibing624/ChatPDF)  PS: 2核CPU 16G内存机器，约2min一条😭
"""

init_message = "欢迎使用 ChatPDF Web UI，可以直接提问或上传文件后提问"


def get_file_list():
    if not os.path.exists("content"):
        return []
    return [f for f in os.listdir("content") if
            f.endswith(".txt") or f.endswith(".pdf") or f.endswith(".docx") or f.endswith(".md")]


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_model", type=str, default="shibing624/text2vec-base-chinese")
    parser.add_argument("--gen_model_type", type=str, default="llama")
    parser.add_argument("--gen_model", type=str, default="LinkSoul/Chinese-Llama-2-7b-4bit")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--int4", action='store_false', help="use int4 quantization")
    parser.add_argument("--int8", action='store_true', help="use int8 quantization")
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=8082)
    parser.add_argument("--share", action='store_true', help="share model")
    args = parser.parse_args()
    logger.info(args)

    model = ChatPDF(
        sim_model_name_or_path=args.sim_model,
        gen_model_type=args.gen_model_type,
        gen_model_name_or_path=args.gen_model,
        lora_model_name_or_path=args.lora_model,
        device=args.device,
        int4=args.int4,
        int8=args.int8,
    )
    file_list = get_file_list()


    def upload_file(file):
        if not os.path.exists(CONTENT_DIR):
            os.mkdir(CONTENT_DIR)
        filename = os.path.basename(file.name)
        file_path = os.path.join(CONTENT_DIR, filename)
        logger.info(f"upload file done: {file_path}")
        shutil.move(file.name, file_path)
        # file_list首位插入新上传的文件
        file_list.insert(0, filename)
        return gr.Dropdown.update(choices=file_list, value=filename)


    def get_answer(query, index_path, history, topn=5, max_input_size=2048):
        if model is None:
            return [None, "模型还未加载"], query
        if index_path:
            if not model.sim_model.corpus_embeddings:
                model.load_index(index_path)
            response, reference_results = model.predict(
                query=query, topn=topn, context_len=max_input_size)
            logger.debug(f"query: {query}, response with content: {response}")
            for i in range(len(reference_results)):
                r = reference_results[i]
                response += f"\n{r.strip()}"
            response = parse_text(response)
            history = history + [[query, response]]
        else:
            logger.debug(f"query: {query}, index: {index_path}, response without content")
            # 未加载文件，仅返回生成模型结果
            model.history.append([query, ''])
            response = ""
            for new_text in model.stream_generate_answer(query, context_len=max_input_size):
                response += new_text
            response = response.strip()
            model.history[-1][1] = response
            response = parse_text(response)
            history = history + [[query, response]]
            logger.debug(f"query: {query}, response: {response}")
        return history, ""


    def update_status(history, status):
        history = history + [[None, status]]
        logger.info(status)
        return history


    def reinit_model(llm_model, embedding_model, history):
        try:
            global model
            if model is not None:
                del model
            model = ChatPDF(
                sim_model_name_or_path=embedding_model_dict.get(
                    embedding_model,
                    "shibing624/text2vec-base-chinese"
                ),
                gen_model_type=llm_model.split('-')[0],
                gen_model_name_or_path=llm_model_dict.get(llm_model, "LinkSoul/Chinese-Llama-2-7b-4bit"),
                lora_model_name_or_path=None,
            )

            model_status = """模型已成功重新加载，请选择文件后点击"加载文件"按钮"""
        except Exception as e:
            model = None
            logger.error(e)
            model_status = """模型未成功重新加载，请重新选择后点击"加载模型"按钮"""
        return history + [[None, model_status]]


    def get_file_hash(fpath):
        return hashlib.md5(open(fpath, 'rb').read()).hexdigest()


    def get_vector_store(filepath, history, embedding_model):
        logger.info(filepath, history)
        index_path = None
        file_status = ''
        if model is not None:
            local_file_path = os.path.join(CONTENT_DIR, filepath)
            local_file_hash = get_file_hash(local_file_path)
            index_file_name = f"{filepath}.{embedding_model}.{local_file_hash}.emb.json"
            local_index_path = os.path.join(CONTENT_DIR, index_file_name)
            if os.path.exists(local_index_path):
                model.load_index(local_index_path)
                index_path = local_index_path
                file_status = "文件已成功加载，请开始提问"
            elif os.path.exists(local_file_path):
                model.load_doc_files(local_file_path)
                model.save_index(local_index_path)
                index_path = local_index_path
                if index_path:
                    file_status = "文件索引并成功加载，请开始提问"
                else:
                    file_status = "文件未成功加载，请重新上传文件"
        else:
            file_status = "模型未完成加载，请先在加载模型后再导入文件"

        return index_path, history + [[None, file_status]]


    def reset_chat(chatbot, state):
        return None, None


    with gr.Blocks(css=block_css) as demo:
        index_path, file_status, model_status = gr.State(""), gr.State(""), gr.State("")
        gr.Markdown(webui_title)
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot([[None, init_message], [None, None]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=700)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入提问内容，按回车进行提交",
                                   ).style(container=False)
                clear_btn = gr.Button('🔄Clear!', elem_id='clear').style(full_width=True)
            with gr.Column(scale=1):
                llm_model = gr.Radio(llm_model_dict_list,
                                     label="LLM 模型",
                                     value=list(llm_model_dict.keys())[0],
                                     interactive=True)
                embedding_model = gr.Radio(embedding_model_dict_list,
                                           label="Embedding 模型",
                                           value=embedding_model_dict_list[0],
                                           interactive=True)
                load_model_button = gr.Button("重新加载模型")

                with gr.Tab("select"):
                    selectFile = gr.Dropdown(
                        file_list,
                        label="content file",
                        interactive=True,
                        value=file_list[0] if len(file_list) > 0 else None
                    )
                with gr.Tab("upload"):
                    file = gr.File(
                        label="content file",
                        file_types=['.txt', '.md', '.docx', '.pdf']
                    )
                load_file_button = gr.Button("加载文件")
        load_model_button.click(
            reinit_model,
            show_progress=True,
            inputs=[llm_model, embedding_model, chatbot],
            outputs=chatbot
        )
        # 将上传的文件保存到content文件夹下,并更新下拉框
        file.upload(upload_file, inputs=file, outputs=selectFile)
        load_file_button.click(
            get_vector_store,
            show_progress=True,
            inputs=[selectFile, chatbot, embedding_model],
            outputs=[index_path, chatbot],
        )
        query.submit(
            get_answer,
            [query, index_path, chatbot],
            [chatbot, query],
        )
        clear_btn.click(reset_chat, [chatbot, query], [chatbot, query])

    demo.queue(concurrency_count=3).launch(
        server_name=args.server_name, server_port=args.server_port, share=args.share
    )
