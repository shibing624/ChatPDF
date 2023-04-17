<div align="right">
  <!-- 语言: -->
  简体中文 | <a title="English" href="./readme/README_en.md">English</a> 
</div>

<h1 align="center">ChatPDF</h1>
<div align="center">
  <a href="https://github.com/shibing624/ChatPDF">
  </a>

<p align="center">
    <h3>为ChatGPT/ChatGLM/LLaMA等多种LLM提供了一个轻快好用的Web图形界面</h3>
    <p align="center">
      <a href="https://github.com/shibing624/ChatPDF/blob/main/LICENSE">
        <img alt="Tests Passing" src="https://img.shields.io/github/license/shibing624/ChatPDF" />
      </a>
      <a href="https://gradio.app/">
        <img alt="GitHub Contributors" src="https://img.shields.io/badge/Base-Gradio-fb7d1a?style=flat" />
      </a>
      <p>
        流式传输 / 无限对话 / 保存对话 / 预设Prompt集 / 联网搜索 / 根据文件回答 <br />
        渲染LaTeX / 渲染表格 / 代码高亮 / 自动亮暗色切换 / 自适应界面 / “小而美”的体验 <br />
        自定义api-Host / 多参数可调 / 多API Key均衡负载 / 多用户显示 / 适配GPT-4 / 支持本地部署LLM
      </p>
      <a href="https://www.bilibili.com/video/BV1184y1w7aP"><strong>介绍视频</strong></a>
	||
      <a href="https://huggingface.co/spaces/shibing624/ChatPDF"><strong>在线体验</strong></a>
      	·
      <a href="https://huggingface.co/login?next=%2Fspaces%2Fshibing624%2FChatPDF%3Fduplicate%3Dtrue"><strong>一键部署</strong></a>
    </p>
    <p align="center">
      <img alt="Animation Demo" src="https://user-images.githubusercontent.com/51039745/226255695-6b17ff1f-ea8d-464f-b69b-a7b6b68fffe8.gif" />
    </p>
  </p>
</div>


## 使用技巧

- 使用System Prompt可以很有效地设定前提条件。
- 使用Prompt模板功能时，选择Prompt模板集合文件，然后从下拉菜单中选择想要的prompt。
- 如果回答不满意，可以使用 `重新生成`按钮再试一次
- 对于长对话，可以使用 `优化Tokens`按钮减少Tokens占用。
- 输入框支持换行，按 `shift enter`即可。
- 可以在输入框按上下箭头在输入历史之间切换
- 部署到服务器：将程序最后一句改成 `demo.launch(server_name="0.0.0.0", server_port=<你的端口号>)`。
- 获取公共链接：将程序最后一句改成 `demo.launch(share=True)`。注意程序必须在运行，才能通过公共链接访问。
- 在Hugging Face上使用：建议在右上角 **复制Space** 再使用，这样App反应可能会快一点。

## 安装方式、使用方式

#### 安装依赖

在终端中输入下面的命令，然后回车即可。
```shell
pip install -r requirements.txt
```

如果你还想使用本地运行大模型的功能，请再执行下面的命令：
```shell
pip install -r requirements_advanced.txt
```

如果您在使用Windows，建议通过WSL，在Linux上安装。如果您没有安装CUDA，并且不想只用CPU跑大模型，请先安装CUDA。

如果下载慢，建议配置豆瓣源。

#### 启动

请使用下面的命令。取决于你的系统，你可能需要用python或者python3命令。请确保你已经安装了Python。
```shell
python ChatPDF.py
```

如果一切顺利，现在，你应该已经可以在浏览器地址栏中输入 http://localhost:7860 查看并使用 ChatPDF 了。
如果您已经有下载好的本地模型了，请将它们放在models文件夹下面（文件名中需要包含llama/alpaca/chatglm），将LoRA模型们放在lora文件夹下面。


## 疑难杂症解决

在遇到各种问题查阅相关信息前，您可以先尝试手动拉取本项目的最新更改并更新 gradio，然后重试。步骤为：

1. 点击网页上的 `Download ZIP` 下载最新代码，或
   ```shell
   git pull https://github.com/shibing624/ChatPDF.git main -f
   ```
2. 尝试再次安装依赖（可能本项目引入了新的依赖）
   ```
   pip install -r requirements.txt
   ```
3. 更新gradio
   ```
   pip install gradio --upgrade --force-reinstall
   ```

很多时候，这样就可以解决问题。

