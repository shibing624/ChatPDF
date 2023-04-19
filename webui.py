# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
modified from https://github.com/shibing624/ChatPDF/blob/master/webui.py
"""
import gradio as gr
import os
import shutil
from loguru import logger
from chatpdf import ChatPDF

VECTOR_SEARCH_TOP_K = 3
MAX_INPUT_LEN = 512

embedding_model_dict = {
    "sentence-transformers": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "shibing624/text2vec-base-chinese",
}

# supported LLM models
llm_model_dict = {
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b-int4-qe": "THUDM/chatglm-6b-int4-qe",
    "chatglm-6b": "THUDM/chatglm-6b",
    "llama-7b": "decapoda-research/llama-7b-hf",
    "llama-13b": "decapoda-research/llama-13b-hf",
}

llm_model_dict_list = list(llm_model_dict.keys())
embedding_model_dict_list = list(embedding_model_dict.keys())

model = ChatPDF(
    sim_model_name_or_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    gen_model_type="chatglm",
    gen_model_name_or_path="THUDM/chatglm-6b-int4",
    lora_model_name_or_path=None,
    max_input_size=MAX_INPUT_LEN,
)


def get_file_list():
    if not os.path.exists("content"):
        return []
    return [f for f in os.listdir("content") if
            f.endswith(".txt") or f.endswith(".pdf") or f.endswith(".docx") or f.endswith(".md")]


file_list = get_file_list()


def upload_file(file):
    os.makedirs("content", exist_ok=True)
    filename = os.path.basename(file.name)
    shutil.move(file.name, "content/" + filename)
    # file_list首位插入新上传的文件
    file_list.insert(0, filename)
    return gr.Dropdown.update(choices=file_list, value=filename)


def get_answer(query, index_path, history):
    if index_path:
        if not model.sim_model.corpus_embeddings:
            model.load_index(index_path)
        response, history = model.query(query, topn=VECTOR_SEARCH_TOP_K)
        history = history + [(query, response)]
    else:
        history = history + [[None, "请先加载文件后，再进行提问。"]]
    return history, ""


def update_status(history, status):
    history = history + [[None, status]]
    logger.info(status)
    return history


def reinit_model(llm_model, embedding_model, history):
    try:
        global model
        del model
        model = ChatPDF(
            sim_model_name_or_path=embedding_model_dict.get(
                embedding_model,
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            gen_model_type=llm_model.split('-')[0],
            gen_model_name_or_path=llm_model_dict.get(llm_model, "THUDM/chatglm-6b-int4"),
            lora_model_name_or_path=None,
            max_input_size=MAX_INPUT_LEN,
        )
        model_status = """模型已成功重新加载，请选择文件后点击"加载文件"按钮"""
    except Exception as e:
        model = None
        logger.error(e)
        model_status = """模型未成功重新加载，请重新选择后点击"加载模型"按钮"""
    return history + [[None, model_status]]


def get_vector_store(filepath, history):
    logger.info(filepath, history)
    if model is not None:
        local_file_path = "content/" + filepath
        local_index_path = "content/" + filepath + ".index.json"
        model.load_pdf_file(local_file_path)
        model.save_index(local_index_path)
        index_path = local_index_path
        if index_path:
            file_status = "文件已成功加载，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        index_path = None
    return index_path, history + [[None, file_status]]


def reset_chat(chatbot, state):
    return None, None


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

Link in: [https://github.com/shibing624/ChatPDF](https://github.com/shibing624/ChatPDF)

"""

init_message = """欢迎使用 ChatPDF Web UI，开始提问前，请依次如下 3 个步骤：
1. 选择语言模型、Embedding 模型及相关参数后点击"重新加载模型"，并等待加载完成提示
2. 上传或选择已有文件作为本地知识文档输入后点击"重新加载文档"，并等待加载完成提示
3. 输入要提交的问题后，点击回车提交 """

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
        inputs=[selectFile, chatbot],
        outputs=[index_path, chatbot],
    )
    query.submit(
        get_answer,
        [query, index_path, chatbot],
        [chatbot, query],
    )
    clear_btn.click(reset_chat, [chatbot, query], [chatbot, query])

demo.queue(concurrency_count=3).launch(
    server_name='0.0.0.0', share=False, inbrowser=False)
