# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""
import argparse
import os

import gradio as gr
from loguru import logger

from chatpdf import ChatPDF

pwd_path = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_model", type=str, default="shibing624/text2vec-base-chinese")
    parser.add_argument("--gen_model_type", type=str, default="auto")
    parser.add_argument("--gen_model", type=str, default="01-ai/Yi-6B-Chat")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--corpus_files", type=str, default="sample.pdf")
    parser.add_argument("--int4", action='store_true', help="use int4 quantization")
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
        corpus_files=args.corpus_files.split(','),
        device=args.device,
        int4=args.int4,
        int8=args.int8,
    )
    logger.info(f"chatpdf model: {model}")


    def predict_stream(message, history):
        history_format = []
        for human, assistant in history:
            history_format.append([human, assistant])
        model.history = history_format
        for chunk in model.predict_stream(message):
            yield chunk


    def predict(message, history):
        logger.debug(message)
        response, reference_results = model.predict(message, do_print=True)
        r = response + "\n\n" + '\n'.join(reference_results)
        logger.debug(r)
        return r


    def vote(data: gr.LikeData):
        if data.liked:
            logger.debug(f"like: {data.value}")
        else:
            logger.debug(f"dislike: {data.value}")


    chatbot_stream = gr.Chatbot(
        height=800,
        avatar_images=(
            os.path.join(pwd_path, "assets/user.png"),
            os.path.join(pwd_path, "assets/llama.png"),
        ), bubble_full_width=False)
    title = " 🎉ChatPDF WebUI🎉 "
    description = "Link in Github: [shibing624/ChatPDF](https://github.com/shibing624/ChatPDF)"
    css = """.toast-wrap { display: none !important } """
    examples = ['Can you tell me about the NLP?', '介绍下NLP']
    chat_interface_stream = gr.ChatInterface(
        predict_stream,
        textbox=gr.Textbox(lines=4, placeholder="Ask me question", scale=7),
        title=title,
        description=description,
        chatbot=chatbot_stream,
        css=css,
        examples=examples,
        theme='soft',
    )

    with gr.Blocks() as demo:
        chatbot_stream.like(vote, None, None)
        chat_interface_stream.render()
    demo.queue().launch(
        server_name=args.server_name, server_port=args.server_port, share=args.share
    )
