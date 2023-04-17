# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from loguru import logger
from similarities import Similarity
from textgen import ChatGlmModel

PROMPT_TEMPLATE = """\
基于以下已知信息，简洁和专业的来回答用户的问题。
如果无法从中得到答案，请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。

已知内容:
{context_str}

问题:
{query_str}
"""


class ChatPDF:
    def __init__(self, pdf_path: str = None, max_input_size: int = 1024, index_path: str = None):
        self.sim_model = Similarity(model_name_or_path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if index_path is not None:
            self.load(index_path)
        elif pdf_path is not None:
            if pdf_path.endswith('.pdf'):
                corpus = self.extract_text_from_pdf(pdf_path)
            elif pdf_path.endswith('.docx'):
                corpus = self.extract_text_from_docx(pdf_path)
            elif pdf_path.endswith('.md'):
                corpus = self.extract_text_from_markdown(pdf_path)
            else:
                corpus = self.extract_text_from_txt(pdf_path)
            self.sim_model.add_corpus(corpus)
        else:
            raise ValueError('pdf_path or index_path must be provided.')
        self.pdf_path = pdf_path
        self.max_input_size = max_input_size
        self.gen_model = ChatGlmModel("chatglm", "THUDM/chatglm-6b-int4")
        self.history = None

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
    def add_source_numbers(lst):
        """Add source numbers to a list of strings."""
        return [f'[{idx + 1}]\t "{item}"' for idx, item in enumerate(lst)]

    def generate_answer(self, query_str, context_str, history=None, max_length=2048):
        """Generate answer from query and context."""
        prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=query_str)
        logger.debug(prompt)
        response, out_history = self.gen_model.chat(prompt, history, max_length=max_length)
        logger.debug(response)
        return response, out_history

    def query(self, query_str, topn=5):
        """Query from corpus."""
        sim_contents = self.sim_model.most_similar(query_str, topn=topn)
        logger.debug(sim_contents)
        reference_results = []
        for query_id, id_score_dict in sim_contents.items():
            for corpus_id, s in id_score_dict.items():
                reference_results.append(self.sim_model.corpus[corpus_id])
        if not reference_results:
            return '没有提供足够的相关信息', reference_results
        reference_results = self.add_source_numbers(reference_results)
        logger.debug(reference_results)
        context_str = '\n'.join(reference_results)[:(self.max_input_size - len(PROMPT_TEMPLATE))]
        response, out_history = self.generate_answer(query_str, context_str, self.history)
        self.history = out_history
        return response, reference_results

    def save(self, path=None):
        """Save model."""
        if path is None:
            path = '.'.join(self.pdf_path.split('.')[:-1]) + '_index.json'
        self.sim_model.save_index(path)

    def load(self, path=None):
        """Load model."""
        if path is None:
            path = '.'.join(self.pdf_path.split('.')[:-1]) + '_index.json'
        self.sim_model.load_index(path)


if __name__ == "__main__":
    m = ChatPDF(pdf_path='sample_paper.pdf')
    print(m.query('自然语言中的非平行迁移是指什么？'))
    print(m.query('文章题目是啥？'))
