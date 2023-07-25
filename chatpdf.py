# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from typing import Union, List

import torch
from loguru import logger
from peft import PeftModel
from similarities import Similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation.utils import GenerationConfig

PROMPT_TEMPLATE = """\
基于以下已知信息，简洁和专业的来回答用户的问题。
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
            device: str = None
    ):
        default_device = 'cpu'
        if torch.cuda.is_available():
            default_device = 'cuda'
        elif torch.backends.mps.is_available():
            default_device = 'mps'
        self.device = device or default_device
        self.sim_model = Similarity(model_name_or_path=sim_model_name_or_path, device=self.device)
        self.gen_model, self.tokenizer = self._init_gen_model(
            gen_model_type,
            gen_model_name_or_path,
            peft_name=lora_model_name_or_path
        )
        self.history = None
        self.doc_files = None

    def _init_gen_model(self, gen_model_type: str, gen_model_name_or_path: str, peft_name: str = None):
        """Init generate model."""
        if gen_model_type == "chatglm":
            model = AutoModel.from_pretrained(
                gen_model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                gen_model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        model.generation_config = GenerationConfig.from_pretrained(gen_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            gen_model_name_or_path,
            use_fast=False,
            trust_remote_code=True
        )
        if peft_name:
            model = PeftModel.from_pretrained(
                model,
                peft_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            logger.info(f"Loaded peft model from {peft_name}")
        return model, tokenizer

    @torch.inference_mode()
    def generate_answer(
            self,
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.0,
            context_len=2048
    ):
        generation_config = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        )
        input_ids = self.tokenizer(prompt).input_ids
        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        generation_output = self.gen_model.generate(
            input_ids=torch.as_tensor([input_ids]).to(self.device),
            **generation_config,
        )
        output_ids = generation_output[0]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        stop_str = self.tokenizer.eos_token
        l_prompt = len(self.tokenizer.decode(input_ids, skip_special_tokens=False))
        pos = output.rfind(stop_str, l_prompt)
        if pos != -1:
            output = output[l_prompt:pos]
        else:
            output = output[l_prompt:]
        return output.strip()

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

    def query(
            self,
            query: str,
            topn: int = 5,
            max_length: int = 512,
            max_input_size: int = 1024,
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

        context_str = '\n'.join(reference_results)[:(max_input_size - len(PROMPT_TEMPLATE))]

        prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=query)
        response = self.generate_answer(prompt, max_new_tokens=max_length)
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
    m = ChatPDF()
    m.load_doc_files(doc_files='sample.pdf')
    response = m.query('自然语言中的非平行迁移是指什么？')
    print(response[0])
    response = m.query('本文作者是谁？')
    print(response[0])
