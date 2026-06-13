GPT-2 训练：
用《斗破苍穹》生成小说文本
用《斗破苍穹》小说文本从头训练 GPT-2 模型（124M 参数）。实现了多卡分布式训练、断点续训，能生成玄幻风格的文本。

能做什么:
给定几个字开头，让模型继续往下写
支持中文输入

主要代码:
train_gpt2.py：训练脚本，包含模型定义、数据加载、DDP 分布式训练、日志记录、保存 checkpoint
predict.py：加载训练好的模型，从输入文字开始生成后续文本

环境要求:
Python 3.10+
至少一张 GPU（推荐 4 张，每张 8GB 以上显存）

安装依赖:
pip install torch>=2.0.0 transformers>=4.30.0 tiktoken modelscope
训练
把小说文本文件放在项目根目录，命名为 doup.txt（UTF-8 编码）

运行：
两张或以上显卡：torchrun --standalone --nproc_per_node=4 train_gpt2.py
如果只有一张显卡：直接 python train_gpt2.py 就可以
训练时会打印 loss、学习率、每秒处理的 token 数，每 50 步保存一次 checkpoint 到 log/ 文件夹

生成文本：
训练完任意一个 checkpoint（例如 log/model_04999.pt）后，修改 predict.py 里的 checkpoint 路径，然后运行
python predict.py
默认以“你是谁？”开头续写 100 个 token，可以修改代码里的 text 变量。

训练效果：
初始 loss 约 6.2，训练 5000 步后降到 1.8
生成的句子有时通顺，偶尔会重复或出现不太通顺的情况，毕竟只跑了 5000 步，而且 BERT 分词器不是专门做生成的
如果想效果好，可以继续训练，有断点续训能力

已知问题：
数据预处理用的是 BertTokenizer，对中文支持尚可，但生成能力不如专用中文模型
分布式训练需用 torchrun 启动，不要直接用 python
断点续训会自动扫描 log/ 下的 model_*.pt 文件，加载最新的那个（脚本里已经写了）
