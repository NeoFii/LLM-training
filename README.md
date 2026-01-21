<div align="center">

# 🚀 LLM Training Learning Journey

记录 NeoFii 的大语言模型训练学习过程

[![GitHub stars](https://img.shields.io/github/stars/NeoFii/LLM-training?style=social)](https://github.com/NeoFii/LLM-training)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Powered by SwanLab](https://img.shields.io/badge/Powered%20by-SwanLab-438440)](https://swanlab.cn/)

[English](README_EN.md) | 简体中文

</div>

## 📖 项目简介

本仓库记录学习大语言模型（LLM）训练的完整过程。
## 📚 学习路线

| 章节 | 主要内容 | 进度 |
|------|----------|------|
| 第一章: 传统模型 | [BERT文本分类](https://docs.swanlab.cn/course/llm_train_course/01-traditionmodel/1.bert/README.html) | Completed |
| 第二章: 预训练 | [从0预训练自己的LLM](https://docs.swanlab.cn/course/llm_train_course/02-pretrain/1.qwen-pretrain/README.html) | Todo |
| 第三章: 微调 | [LoRA指令微调Qwen3-4B-Base](https://docs.swanlab.cn/course/llm_train_course/03-sft/3.glm4-instruct/README.html) | Todo |
| 第四章: 强化学习 | [GRPO训练模型玩数独](https://docs.swanlab.cn/course/llm_train_course/04-reinforce/3.sudoku_grpo/README.html) | Todo |
| 第五章: 评测 | [基于EvalScope模型评测](https://docs.swanlab.cn/course/llm_train_course/05-eval/1.evalscope/README.html) | Todo |
| 第六章: 语音模型 | [CosyVoice2实现派蒙语音的微调](https://docs.swanlab.cn/course/llm_train_course/07-audio/1.cosyvoice-sft/README.html) | Todo |

## 📊 实验追踪

本项目使用 [SwanLab](https://swanlab.cn/) 进行实验追踪和可视化。所有训练过程、指标变化都会被完整记录。
```python
import swanlab

# 初始化实验
swanlab.init(
    project="llm-training-learning",
    experiment_name="chapter2-pretrain"
)
```

## 🙏 致谢

特别感谢 [SwanLab](https://github.com/SwanHubX/SwanLab) 为本项目提供支持！

SwanLab 是一个开源、轻量的 AI 模型训练追踪与可视化工具，提供跟踪、记录、比较和协作实验的平台。

- 官方网站：[https://swanlab.cn/](https://swanlab.cn/)
- GitHub：[https://github.com/SwanHubX/SwanLab](https://github.com/SwanHubX/SwanLab)

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

<div align="center">

**⭐ 如果这个项目对你有帮助，欢迎 Star ⭐**

Made with ❤️ by NeoFii

</div>