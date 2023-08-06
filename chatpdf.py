# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
from threading import Thread
from typing import Union, List

import torch
from loguru import logger
from peft import PeftModel
from similarities import Similarity
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BloomForCausalLM,
    BloomTokenizerFast,
    LlamaTokenizer,
    LlamaForCausalLM,
    TextIteratorStreamer,
    GenerationConfig,
)

MODEL_CLASSES = {
    "bloom": (BloomForCausalLM, BloomTokenizerFast),
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "baichuan": (AutoModelForCausalLM, AutoTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer),
}

PROMPT_TEMPLATE = """基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

已知内容:
{context_str}

问题:
{query_str}
"""


class ChatPDF:
    def __init__(
            self,
            sim_model_name_or_path: str = "shibing624/text2vec-base-chinese",
            gen_model_type: str = "baichuan",
            gen_model_name_or_path: str = "baichuan-inc/Baichuan-13B-Chat",
            lora_model_name_or_path: str = None,
            device: str = None,
            int8: bool = False,
            int4: bool = False,
    ):
        default_device = torch.device('cpu')
        if torch.cuda.is_available():
            default_device = torch.device(0)
        elif torch.backends.mps.is_available():
            default_device = 'mps'
        self.device = device or default_device
        self.sim_model = Similarity(model_name_or_path=sim_model_name_or_path, device=self.device)
        self.gen_model, self.tokenizer = self._init_gen_model(
            gen_model_type,
            gen_model_name_or_path,
            peft_name=lora_model_name_or_path,
            int8=int8,
            int4=int4,
        )
        self.history = []
        self.doc_files = None

    def _init_gen_model(
            self,
            gen_model_type: str,
            gen_model_name_or_path: str,
            peft_name: str = None,
            int8: bool = False,
            int4: bool = False,
    ):
        """Init generate model."""
        if int8 or int4:
            device_map = None
        else:
            device_map = "auto"
        model_class, tokenizer_class = MODEL_CLASSES[gen_model_type]
        tokenizer = tokenizer_class.from_pretrained(gen_model_name_or_path, trust_remote_code=True)
        model = model_class.from_pretrained(
            gen_model_name_or_path,
            load_in_8bit=int8 if gen_model_type not in ['baichuan', 'chatglm'] else False,
            load_in_4bit=int4 if gen_model_type not in ['baichuan', 'chatglm'] else False,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True,
        )
        if self.device == torch.device('cpu'):
            model.float()
        if gen_model_type in ['baichuan', 'chatglm']:
            if int4:
                model = model.quantize(4).cuda()
            elif int8:
                model = model.quantize(8).cuda()
        try:
            model.generation_config = GenerationConfig.from_pretrained(gen_model_name_or_path, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Failed to load generation config from {gen_model_name_or_path}, {e}")
        if peft_name:
            model = PeftModel.from_pretrained(
                model,
                peft_name,
                torch_dtype=torch.float16,
            )
            logger.info(f"Loaded peft model from {peft_name}")
        model.eval()
        return model, tokenizer

    @torch.inference_mode()
    def stream_generate_answer(
            self,
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            repetition_penalty=1.0,
            context_len=2048
    ):
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        input_ids = self.tokenizer(prompt).input_ids
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        generation_kwargs = dict(
            input_ids=torch.as_tensor([input_ids]).to(self.device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )
        thread = Thread(target=self.gen_model.generate, kwargs=generation_kwargs)
        thread.start()

        yield from streamer

    def load_doc_files(self, doc_files: Union[str, List[str]]):
        """Load document files."""
        if isinstance(doc_files, str):
            doc_files = [doc_files]
        for doc_file in doc_files:
            if doc_file.endswith('.pdf'):
                corpus = self.extract_text_from_pdf(doc_file)
            elif doc_file.endswith('.docx'):
                corpus = self.extract_text_from_docx(doc_file)
            elif doc_file.endswith('.md'):
                corpus = self.extract_text_from_markdown(doc_file)
            else:
                corpus = self.extract_text_from_txt(doc_file)
            self.sim_model.add_corpus(corpus)
        self.doc_files = doc_files

    @staticmethod
    def extract_text_from_pdf(file_path: str):
        """Extract text content from a PDF file."""
        import PyPDF2
        contents = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text().strip()
                raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
                new_text = ''
                for text in raw_text:
                    new_text += text
                    if text[-1] in ['.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》', '」',
                                    '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']:
                        contents.append(new_text)
                        new_text = ''
                if new_text:
                    contents.append(new_text)
        return contents

    @staticmethod
    def extract_text_from_txt(file_path: str):
        """Extract text content from a TXT file."""
        contents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = [text.strip() for text in f.readlines() if text.strip()]
        return contents

    @staticmethod
    def extract_text_from_docx(file_path: str):
        """Extract text content from a DOCX file."""
        import docx
        document = docx.Document(file_path)
        contents = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        return contents

    @staticmethod
    def extract_text_from_markdown(file_path: str):
        """Extract text content from a Markdown file."""
        import markdown
        from bs4 import BeautifulSoup
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        html = markdown.markdown(markdown_text)
        soup = BeautifulSoup(html, 'html.parser')
        contents = [text.strip() for text in soup.get_text().splitlines() if text.strip()]
        return contents

    @staticmethod
    def _add_source_numbers(lst):
        """Add source numbers to a list of strings."""
        return [f'[{idx + 1}]\t "{item}"' for idx, item in enumerate(lst)]

    def predict(
            self,
            query: str,
            topn: int = 5,
            max_length: int = 512,
            context_len: int = 2048,
            temperature: float = 0.7,
            do_print: bool = True,
    ):
        """Query from corpus."""

        sim_contents = self.sim_model.most_similar(query, topn=topn)

        reference_results = []
        for query_id, id_score_dict in sim_contents.items():
            for corpus_id, s in id_score_dict.items():
                reference_results.append(self.sim_model.corpus[corpus_id])
        if not reference_results:
            return '没有提供足够的相关信息', reference_results
        reference_results = self._add_source_numbers(reference_results)
        context_str = '\n'.join(reference_results)[:(context_len - len(PROMPT_TEMPLATE))]

        prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=query)
        self.history.append([prompt, ''])
        response = ""
        if do_print:
            print(f"> ", end="", flush=True)
        for new_text in self.stream_generate_answer(
                self.gen_model,
                self.tokenizer,
                prompt,
                self.device,
                max_new_tokens=max_length,
                temperature=temperature,
                context_len=context_len,
        ):
            response += new_text
            if do_print:
                print(new_text, end="", flush=True)
        if do_print:
            print()
        response = response.strip()
        self.history[-1][1] = response
        return response, reference_results

    def save_index(self, index_path=None):
        """Save model."""
        if index_path is None:
            index_path = '.'.join(self.doc_files.split('.')[:-1]) + '_index.json'
        self.sim_model.save_index(index_path)

    def load_index(self, index_path=None):
        """Load model."""
        if index_path is None:
            index_path = '.'.join(self.doc_files.split('.')[:-1]) + '_index.json'
        self.sim_model.load_index(index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_model", type=str, default="shibing624/text2vec-base-chinese")
    parser.add_argument("--gen_model_type", type=str, default="baichuan")
    parser.add_argument("--gen_model", type=str, default="baichuan-inc/Baichuan-13B-Chat")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--int4", action='store_true', help="use int4 quantization")
    parser.add_argument("--int8", action='store_true', help="use int8 quantization")
    args = parser.parse_args()
    print(args)
    m = ChatPDF(
        sim_model_name_or_path=args.sim_model,
        gen_model_type=args.gen_model_type,
        gen_model_name_or_path=args.gen_model,
        lora_model_name_or_path=args.lora_model,
        device=args.device,
        int4=args.int4,
        int8=args.int8
    )
    m.load_doc_files(doc_files='sample.pdf')
    m.predict('自然语言中的非平行迁移是指什么？', do_print=True)
    m.predict('本文作者是谁？', do_print=True)
