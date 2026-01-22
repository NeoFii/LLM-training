"""
用预训练的Bert模型微调IMDB数据集，并使用SwanLabCallback回调函数将结果上传到SwanLab。
IMDB数据集的1是positive，0是negative。
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from swanlab.integration.transformers import SwanLabCallback
import swanlab

def predict(text, model, tokenizer, CLASS_NAME):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs) # bert是encoder-only的模型 预测结果时不需要调用 model.generate()函数来生成结果 本身做分类任务 会输出在每一个类别上的logit分数 最终通过argmax找到类别最大的索引即代表其分类
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()

    print(f"Input Text: {text}")
    print(f"Predicted class: {int(predicted_class)} {CLASS_NAME[int(predicted_class)]}")
    return int(predicted_class)

# 加载IMDB数据集
dataset = load_dataset('imdb', cache_dir="./data")

# 加载预训练的BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 定义tokenize函数
def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True) # padding对长度不同的序列用0填充 便于批处理; truncation允许tokenize时对长度超出处理范围的序列进行切割

# 对数据集进行tokenization batched=True使用批量处理加快处理速度
tokenized_datasets = dataset.map(tokenize, batched=True)

# 设置模型输入格式
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# 加载预训练的BERT模型
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=5,
    # 总的训练轮数
    num_train_epochs=3,
    weight_decay=0.01,
    report_to="none",
    # 单卡训练
)

CLASS_NAME = {0: "negative", 1: "positive"}

# 设置swanlab回调函数
swanlab_callback = SwanLabCallback(project='BERT',
                                   experiment_name='BERT-IMDB',
                                   config={'dataset': 'IMDB', "CLASS_NAME": CLASS_NAME})

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    callbacks=[swanlab_callback],
)

# 训练模型
trainer.train()

# 保存模型
model.save_pretrained('./sentiment_model')
tokenizer.save_pretrained('./sentiment_model')


swanlab.finish()