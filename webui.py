# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: pip install gradio==5.22.0
"""
import argparse

import gradio as gr
from loguru import logger

from rag import Rag

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model_type", type=str, default="auto")
    parser.add_argument("--gen_model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--rerank_model_name", type=str, default="")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--corpus_files", type=str, default="data/sample.pdf")
    parser.add_argument("--int4", action='store_true', help="use int4 quantization")
    parser.add_argument("--int8", action='store_true', help="use int8 quantization")
    parser.add_argument("--chunk_size", type=int, default=220)
    parser.add_argument("--chunk_overlap", type=int, default=0)
    parser.add_argument("--num_expand_context_chunk", type=int, default=1)
    parser.add_argument("--server_name", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", type=int, default=8082)
    parser.add_argument("--share", action='store_true', help="share model")
    args = parser.parse_args()
    logger.info(args)

    model = Rag(
        generate_model_type=args.gen_model_type,
        generate_model_name_or_path=args.gen_model_name,
        lora_model_name_or_path=args.lora_model,
        corpus_files=args.corpus_files.split(','),
        device=args.device,
        int4=args.int4,
        int8=args.int8,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        num_expand_context_chunk=args.num_expand_context_chunk,
        rerank_model_name_or_path=args.rerank_model_name,
    )
    logger.info(f"chatpdf model: {model}")


    def predict_stream(message, history):
        history_format = []
        for item in history:
            if isinstance(item, dict):
                # OpenAI格式
                if item["role"] == "user":
                    if len(history_format) > 0 and len(history_format[-1]) == 1:
                        # 上一条是用户消息但没有回复，添加回复
                        history_format[-1].append("")
                    history_format.append([item["content"]])
                elif item["role"] == "assistant" and len(history_format) > 0:
                    # 助手回复
                    if len(history_format[-1]) == 1:
                        history_format[-1].append(item["content"])
                    else:
                        # 如果上一条已经有回复，创建新条目
                        history_format.append(["", item["content"]])
            else:
                # 兼容旧格式
                history_format.append(item)

        model.history = history_format

        # 跟踪生成的内容以便检索引用结果
        response_text = ""
        for chunk in model.predict_stream(message):
            response_text += chunk
            yield chunk


    chat_interface = gr.ChatInterface(
        fn=predict_stream,
        title=" 🎉ChatPDF WebUI🎉 ",
        description="Link in Github: [shibing624/ChatPDF](https://github.com/shibing624/ChatPDF)",
        examples=['Can you tell me about the NLP?', '介绍下NLP'],
        type="messages",
        textbox=gr.Textbox(
            lines=4,
            placeholder="Ask me question",
        ),
    )
    chat_interface.queue()
    chat_interface.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )
