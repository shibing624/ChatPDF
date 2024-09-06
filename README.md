<h1 align="center">ChatPDF</h1>
<div align="center">
  <a href="https://github.com/shibing624/ChatPDF">
  </a>

<p align="center">
    <h3>基于本地 LLM 做检索知识问答(RAG)</h3>
    <p align="center">
      <a href="https://github.com/shibing624/ChatPDF/blob/main/LICENSE">
        <img alt="Tests Passing" src="https://img.shields.io/github/license/shibing624/ChatPDF" />
      </a>
      <a href="https://gradio.app/">
        <img alt="GitHub Contributors" src="https://img.shields.io/badge/Base-Gradio-fb7d1a?style=flat" />
      </a>
      <p>
        根据文件回答 / 开源模型 / 本地部署LLM
      </p>
    </p>
    <p align="center">
      <img alt="Animation Demo" src="https://github.com/shibing624/ChatPDF/blob/main/docs/snap.png" width="860" />
    </p>
  </p>
</div>

- 本项目实现了轻量版的GraphRAG
  - 支持`local`模式的关系图检索的文档问答
  - 支持Openai API, Deepseek API, Ollama API等，可自行扩展支持更多LLM
  - 支持openai embedding、本地 text2vec embedding、huggingface embedding、sentence-transformers embedding等
  - 异步开发，支持多个API并发请求
- 本项目支持多种开源LLM模型，包括ChatGLM3-6b、Chinese-LLaMA-Alpaca-2、Baichuan、YI等
- 本项目支持多种文件格式，包括PDF、docx、markdown、txt等
- 本项目优化了RAG准确率
  - Chinese chunk切分优化，适配中英文混合文档
  - embedding优化，使用text2vec的sentence embedding，支持sentence embedding/字面相似度匹配算法
  - 检索匹配优化，引入jieba分词的rank_BM25，提升对query关键词的字面匹配，使用字面相似度+sentence embedding向量相似度加权获取corpus候选集
  - 新增reranker模块，对字面+语义检索的候选集进行rerank排序，减少候选集，并提升候选命中准确率，用`rerank_model_name_or_path`参数设置rerank模型
  - 新增候选chunk扩展上下文功能，用`num_expand_context_chunk`参数设置命中的候选chunk扩展上下文窗口大小
  - RAG底模优化，可以使用200k的基于RAG微调的LLM模型，支持自定义RAG模型，用`generate_model_name_or_path`参数设置底模
- 本项目基于gradio开发了RAG对话页面，支持流式对话

## 原理

<img src="https://github.com/shibing624/ChatPDF/blob/main/docs/chatpdf.jpg" width="860" />

## Usage

### 安装依赖

在终端中输入下面的命令，然后回车即可。
```shell
pip install -r requirements.txt
```

如果您在使用Windows，建议通过WSL，在Linux上安装。如果您没有安装CUDA，并且不想只用CPU跑大模型，请先安装CUDA。

如果下载慢，建议配置豆瓣源。

### RAG示例

请使用下面的命令。取决于你的系统，你可能需要用python或者python3命令。请确保你已经安装了Python。
```shell
CUDA_VISIBLE_DEVICES=0 python rag.py
```



### 启动Gradio的Web服务

```shell
CUDA_VISIBLE_DEVICES=0 python webui.py --corpus_files data/sample.pdf --share
```

现在，你应该已经可以在浏览器地址栏中输入 http://localhost:7860 查看并使用 ChatPDF 了。

### GraphRAG示例
> [!TIP]
>
>  **Please set OpenAI API key in environment: `export OPENAI_API_KEY="sk-..."`.** 
>
> If you don't have LLM key, check out this [graphrag._model.py](https://github.com/shibing624/ChatPDF/blob/main/graphrag/_model.py#L120) that using `ollama` .

```shell
python graphrag_demo.py
```


## Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/ChatPDF.svg)](https://github.com/shibing624/ChatPDF/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。

<img src="https://github.com/shibing624/ChatPDF/blob/main/docs/wechat.jpeg" width="200" />

<img src="https://github.com/shibing624/ChatPDF/blob/main/docs/wechat_group.jpg" width="200" />

## License


授权协议为 [The Apache License 2.0](LICENSE)，可免费用做商业用途。请在产品说明中附加ChatPDF的链接和授权协议。


## Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目。

### 关联项目推荐
- [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT)：训练自己的GPT大模型，实现了包括增量预训练、有监督微调、RLHF(奖励建模、强化学习训练)和DPO(直接偏好优化)
