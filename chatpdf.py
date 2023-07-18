# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from typing import Union, List

import torch
from similarities import Similarity
from textgen import ChatGlmModel, GptModel

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
            sim_model_name_or_path: str = "shibing624/text2vec-base-multilingual",
            gen_model_type: str = "chatglm",
            gen_model_name_or_path: str = "THUDM/chatglm-6b-int4",
            lora_model_name_or_path: str = None,

    ):
        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        self.sim_model = Similarity(model_name_or_path=sim_model_name_or_path, device=device)

        if gen_model_type == "chatglm":
            self.gen_model = ChatGlmModel(gen_model_type, gen_model_name_or_path, peft_name=lora_model_name_or_path)
        else:
            self.gen_model = GptModel(gen_model_type, gen_model_name_or_path, peft_name=lora_model_name_or_path)
        self.history = None
        self.doc_files = None

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

    def _generate_answer(self, query_str: str, context_str: str, history=None, max_length=1024):
        """Generate answer from query and context."""
        prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=query_str)
        response, out_history = self.gen_model.chat(prompt, history, max_length=max_length)
        return response, out_history

    def query(
            self,
            query: str,
            topn: int = 5,
            max_length: int = 1024,
            max_input_size: int = 1024,
            use_history: bool = False
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

        if use_history:
            response, out_history = self._generate_answer(query, context_str, self.history, max_length=max_length)
            self.history = out_history
        else:

            response, out_history = self._generate_answer(query, context_str)

        return response, out_history, reference_results

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
    m = ChatPDF(gen_model_name_or_path="THUDM/chatglm-6b-int4-qe")
    m.load_doc_files(doc_files='sample.pdf')
    response = m.query('自然语言中的非平行迁移是指什么？')
    print(response[0])
    response = m.query('本文作者是谁？')
    print(response[0])
