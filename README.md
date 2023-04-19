<h1 align="center">ChatPDF</h1>
<div align="center">
  <a href="https://github.com/shibing624/ChatPDF">
  </a>

<p align="center">
    <h3>为ChatGLM/LLaMA等多种LLM提供了一个好用的基于PDF问答的图形界面</h3>
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
      <a href="https://huggingface.co/spaces/shibing624/ChatPDF"><strong>在线体验</strong></a>
      	·
      <a href="https://huggingface.co/login?next=%2Fspaces%2Fshibing624%2FChatPDF%3Fduplicate%3Dtrue"><strong>一键部署</strong></a>
    </p>
    <p align="center">
      <img alt="Animation Demo" src="https://github.com/shibing624/ChatPDF/blob/main/img.jpg" />
    </p>
  </p>
</div>


## 使用说明

#### 安装依赖

在终端中输入下面的命令，然后回车即可。
```shell
pip install -r requirements.txt
```

如果您在使用Windows，建议通过WSL，在Linux上安装。如果您没有安装CUDA，并且不想只用CPU跑大模型，请先安装CUDA。

如果下载慢，建议配置豆瓣源。

#### 启动

请使用下面的命令。取决于你的系统，你可能需要用python或者python3命令。请确保你已经安装了Python。
```shell
python chatpdf.py
```

如果一切顺利，现在，你应该已经可以在浏览器地址栏中输入 http://localhost:7860 查看并使用 ChatPDF 了。
如果您已经有下载好的本地模型了，设置`model_name_or_path`为对应的模型文件夹即可。


## Contact

- Issue(建议)：[![GitHub issues](https://img.shields.io/github/issues/shibing624/ChatPDF.svg)](https://github.com/shibing624/ChatPDF/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我：加我*微信号：xuming624, 备注：姓名-公司-NLP* 进NLP交流群。


## License


授权协议为 [The Apache License 2.0](LICENSE)，可免费用做商业用途。请在产品说明中附加ChatPDF的链接和授权协议。


## Contribute
项目代码还很粗糙，如果大家对代码有所改进，欢迎提交回本项目。

## Reference
- [imClumsyPanda/langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)
