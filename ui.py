# -*- coding: utf-8 -*-
import os
import argparse
from constants import APP_VERSION,APP_NAME

import gradio as gr
from loguru import logger

from chat import Chat

pwd_path = os.path.abspath(os.path.dirname(__file__))

def _get_footer_message() -> str:
    version = f"<center><p> {APP_VERSION} "
    footer_msg = version + (
        '  © 2023 - 2024 <a href="https://github.com/rupeshs">'
        " Rupesh Sreeraman</a></p></center>"
    )
    return footer_msg

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model_type", type=str, default="auto")
    parser.add_argument("--gen_model_name", type=str, default="01-ai/Yi-6B-Chat")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--rerank_model_name", type=str, default="maidalun1020/bce-reranker-base_v1")
    parser.add_argument("--device", type=str, default=None)
    # parser.add_argument("--corpus_files", type=str, default="sample.pdf")
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

    model = Chat(
        generate_model_type=args.gen_model_type,
        generate_model_name_or_path=args.gen_model_name,
        lora_model_name_or_path=args.lora_model,
        # corpus_files=args.corpus_files.split(','),
        device=args.device,
        int4=args.int4,
        int8=args.int8,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        num_expand_context_chunk=args.num_expand_context_chunk,
        rerank_model_name_or_path=args.rerank_model_name,
    )

    logger.info(f"chat model: {model}")

    def predict_stream(message, history):
        history_format = []
        for human, assistant in history:
            history_format.append([human, assistant])
        model.history = history_format
        for chunk in model.predict_stream(message):
            yield chunk

    def predict(message, history):
        logger.debug(message)
        response, reference_results = model.predict(message)
        r = response + "\n\n" + '\n'.join(reference_results)
        logger.debug(r)
        return r
        
    chatbot_stream = gr.Chatbot(
        height=600,
        avatar_images=(
            os.path.join(pwd_path, "assets/user.png"),
            os.path.join(pwd_path, "assets/llama.png"),
        ), bubble_full_width=False)
    description = "Link in Github: [shibing624/ChatPDF](https://github.com/shibing624/ChatPDF)"
    css = """.toast-wrap { display: none !important } """
    examples = ['Can you tell me about the NLP?', '介绍下NLP']

    chat_interface_stream = gr.ChatInterface(
        predict_stream,
        textbox=gr.Textbox(lines=4, placeholder="问个问题吧！", scale=7),
        description=description,
        chatbot=chatbot_stream,
        css=css,
        examples=examples,
        theme='soft',
    )

    def get_web_ui() -> gr.Blocks:
        def change_mode(mode):
            print(mode)
            # global app_settings
            # app_settings.settings.lcm_diffusion_setting.use_lcm_lora = False
            # app_settings.settings.lcm_diffusion_setting.use_openvino = False
            # if mode == "LCM-LoRA":
            #     app_settings.settings.lcm_diffusion_setting.use_lcm_lora = True
            # elif mode == "LCM-OpenVINO":
            #     app_settings.settings.lcm_diffusion_setting.use_openvino = True


        with gr.Blocks(
            title=APP_NAME,
        ) as fastsd_web_ui:
            gr.HTML("<center><H1>"+APP_NAME+"</H1></center>")
            current_mode = "正常对话"

            mode = gr.Radio(
                ["正常对话", "文档对话", "所有文档对话"],
                label="模型",
                info="当前使用的模型：",
                value=current_mode,
            )
            mode.change(change_mode, inputs=mode)

            chat_interface_stream.render()

            # with gr.Tabs():
            #     with gr.TabItem("文字生成图片"):
            #         get_text_to_image_ui()
            #     with gr.TabItem("图片生成图片"):
            #         get_image_to_image_ui()

            # gr.HTML(_get_footer_message())

        return fastsd_web_ui

    def start_webui(
        share: bool = True,
    ):
        webui = get_web_ui()
        webui.queue().launch(share=share,server_name='0.0.0.0')

    print("Starting web UI mode")
    start_webui(share=True
    )