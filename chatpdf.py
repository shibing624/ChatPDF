# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import argparse
import hashlib
import os
import textwrap
from threading import Thread
from typing import Union, List

import jieba
import torch
from loguru import logger
from peft import PeftModel
from similarities import EnsembleSimilarity, BertSimilarity, BM25Similarity
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


class ChineseTextSplitter:
    def __init__(self, chunk_size=250, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        if any("\u4e00" <= ch <= "\u9fff" for ch in text):
            # check if contains chinese characters
            chunks = jieba.lcut(text)
            chunks = [''.join(chunks[i:i + self.chunk_size]) for i in
                      range(0, len(chunks), self.chunk_size - self.chunk_overlap)]
        else:
            chunks = textwrap.wrap(text, width=self.chunk_size)

        return chunks


class ChatPDF:
    def __init__(
            self,
            sim_model_name_or_path: str = "shibing624/text2vec-base-chinese",
            gen_model_type: str = "baichuan",
            gen_model_name_or_path: str = "baichuan-inc/Baichuan-13B-Chat",
            lora_model_name_or_path: str = None,
            corpus_files: Union[str, List[str]] = None,
            save_corpus_emb_dir: str = "./corpus_embs/",
            device: str = None,
            int8: bool = False,
            int4: bool = False,
            chunk_size: int = 250,
            chunk_overlap: int = 50,
    ):
        default_device = torch.device('cpu')
        if torch.cuda.is_available():
            default_device = torch.device(0)
        elif torch.backends.mps.is_available():
            default_device = 'mps'
        self.device = device or default_device
        self.text_splitter = ChineseTextSplitter(chunk_size, chunk_overlap)
        m1 = BertSimilarity(model_name_or_path=sim_model_name_or_path, device=self.device)
        m2 = BM25Similarity()
        self.sim_model = EnsembleSimilarity(similarities=[m1, m2], weights=[0.5, 0.5], c=2)
        self.gen_model, self.tokenizer = self._init_gen_model(
            gen_model_type,
            gen_model_name_or_path,
            peft_name=lora_model_name_or_path,
            int8=int8,
            int4=int4,
        )
        self.history = []
        self.corpus_files = corpus_files
        if corpus_files:
            self.add_corpus(corpus_files)
        self.save_corpus_emb_dir = save_corpus_emb_dir

    def __str__(self):
        return f"Similarity model: {self.sim_model}, Generate model: {self.gen_model}"

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

    def _get_chat_input(self):
        messages = []
        for conv in self.history:
            if conv and len(conv) > 0 and conv[0]:
                messages.append({'role': 'user', 'content': conv[0]})
            if conv and len(conv) > 1 and conv[1]:
                messages.append({'role': 'assistant', 'content': conv[1]})
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt'
        )
        return input_ids.to(self.gen_model.device)

    @torch.inference_mode()
    def stream_generate_answer(
            self,
            max_new_tokens=512,
            temperature=0.7,
            repetition_penalty=1.0,
            context_len=2048
    ):
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        input_ids = self._get_chat_input()
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        generation_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )
        thread = Thread(target=self.gen_model.generate, kwargs=generation_kwargs)
        thread.start()

        yield from streamer

    def add_corpus(self, files: Union[str, List[str]]):
        """Load document files."""
        if isinstance(files, str):
            files = [files]
        for doc_file in files:
            if doc_file.endswith('.pdf'):
                corpus = self.extract_text_from_pdf(doc_file)
            elif doc_file.endswith('.docx'):
                corpus = self.extract_text_from_docx(doc_file)
            elif doc_file.endswith('.md'):
                corpus = self.extract_text_from_markdown(doc_file)
            else:
                corpus = self.extract_text_from_txt(doc_file)
            full_text = ' '.join(corpus)
            chunks = self.text_splitter.split_text(full_text)
            self.sim_model.add_corpus(chunks)
        self.corpus_files = files

    @staticmethod
    def get_file_hash(fpaths):
        hasher = hashlib.md5()
        target_file_data = bytes()
        if isinstance(fpaths, str):
            fpaths = [fpaths]
        for fpath in fpaths:
            with open(fpath, 'rb') as file:
                chunk = file.read(1024 * 1024)  # read only first 1MB
                hasher.update(chunk)
                target_file_data += chunk

        hash_name = hasher.hexdigest()[:32]
        return hash_name

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

    def predict_stream(
            self,
            query: str,
            topn: int = 5,
            max_length: int = 512,
            context_len: int = 2048,
            temperature: float = 0.7,
    ):
        """Generate predictions stream."""
        reference_results = []
        stop_str = self.tokenizer.eos_token if self.tokenizer.eos_token else "</s>"
        if self.sim_model.corpus:
            logger.debug(f"corpus size: {len(self.sim_model.corpus)}, top3: {list(self.sim_model.corpus.values())[:3]}")
            sim_contents = self.sim_model.most_similar(query, topn=topn)
            # Get reference results
            for query_id, id_score_dict in sim_contents.items():
                for corpus_id, s in id_score_dict.items():
                    reference_results.append(self.sim_model.corpus[corpus_id])
            if not reference_results:
                yield '没有提供足够的相关信息', reference_results
            reference_results = self._add_source_numbers(reference_results)
            context_str = '\n'.join(reference_results)[:(context_len - len(PROMPT_TEMPLATE))]
            prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=query)
            logger.debug(prompt)
        else:
            prompt = query
            logger.debug(prompt)
        self.history.append([prompt, ''])
        response = ""
        for new_text in self.stream_generate_answer(
                max_new_tokens=max_length,
                temperature=temperature,
                context_len=context_len,
        ):
            if new_text != stop_str:
                response += new_text
                yield response

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
        reference_results = []
        if self.sim_model.corpus:
            logger.debug(f"corpus size: {len(self.sim_model.corpus)}, top3: {list(self.sim_model.corpus.values())[:3]}")
            sim_contents = self.sim_model.most_similar(query, topn=topn)
            # Get reference results
            for query_id, id_score_dict in sim_contents.items():
                for corpus_id, s in id_score_dict.items():
                    reference_results.append(self.sim_model.corpus[corpus_id])
            if not reference_results:
                return '没有提供足够的相关信息', reference_results
            reference_results = self._add_source_numbers(reference_results)
            context_str = '\n'.join(reference_results)[:(context_len - len(PROMPT_TEMPLATE))]
            logger.debug(f"context_str: {context_str}")
            prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=query)
        else:
            prompt = query
        self.history.append([prompt, ''])
        response = ""
        for new_text in self.stream_generate_answer(
                max_new_tokens=max_length,
                temperature=temperature,
                context_len=context_len,
        ):
            response += new_text
            if do_print:
                print(new_text, end="", flush=True)
        if do_print:
            print("", flush=True)
        response = response.strip()
        self.history[-1][1] = response
        return response, reference_results

    def save_corpus_emb(self):
        dir_name = self.get_file_hash(self.corpus_files)
        save_dir = os.path.join(self.save_corpus_emb_dir, dir_name)
        self.sim_model.save_corpus_embeddings(save_dir)
        return save_dir

    def load_corpus_emb(self, emb_dir: str):
        logger.debug(f"Loading corpus embeddings from {emb_dir}")
        self.sim_model.load_corpus_embeddings(emb_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_model", type=str, default="shibing624/text2vec-base-chinese")
    parser.add_argument("--gen_model_type", type=str, default="baichuan")
    parser.add_argument("--gen_model", type=str, default="baichuan-inc/Baichuan-13B-Chat")
    parser.add_argument("--lora_model", type=str, default=None)
    parser.add_argument("--corpus_files", type=str, default="sample.pdf")
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
        int8=args.int8,
        save_corpus_emb_dir='./corpus_embs/',
    )
    m.predict('自然语言中的非平行迁移是指什么？', do_print=True)
    files = args.corpus_files.split(',')
    m.add_corpus(files=files)
    m.predict('自然语言中的非平行迁移是指什么？', do_print=True)
    save_dir = m.save_corpus_emb()
    del m
    m = ChatPDF(
        sim_model_name_or_path=args.sim_model,
        gen_model_type=args.gen_model_type,
        gen_model_name_or_path=args.gen_model,
        lora_model_name_or_path=args.lora_model,
        device=args.device,
        int4=args.int4,
        int8=args.int8,
        save_corpus_emb_dir='./corpus_embs/',
    )
    m.load_corpus_emb(save_dir)
    m.predict('自然语言中的非平行迁移是指什么？', do_print=True)
