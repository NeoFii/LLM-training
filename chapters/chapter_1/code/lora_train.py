# LoRA fine tuning bert for classification by peft

from peft import LoraConfig, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import swanlab
from swanlab.integration.transformers import SwanLabCallback


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


dataset = load_dataset('imdb', cache_dir="./data") # 读进来的数据是Python数据类型

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", trust_remote_code=True)

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True)

tokenized_datasets = dataset.map(tokenize, batched=True)

# hf的Trainer中写死了传递的label名称必须为labels 所以需要转换
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value", "attention.output.dense"]
)

# PeftModel
model = get_peft_model(model, lora_config)

training_arguments = TrainingArguments(
    output_dir="./bert_lora/",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=10,
    num_train_epochs=5,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=1000,

    # 添加学习率调度
    lr_scheduler_type="cosine",  # 余弦退火
    warmup_ratio=0.1,  # 前10%进行warmup
    report_to="none"
)

global_config = {
    "lora": lora_config.to_dict(),
    "training": training_arguments.to_dict(),
    "model": {
        "name": "bert-base-uncased",
        "num_labels": 2,
    }
}

swanlab_callback = SwanLabCallback(
    project="bert_lora",
    experiment_name="baseline_epoch5_warmup+cosine_r32_qkvo",
    config=global_config
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    callbacks=[swanlab_callback],
    compute_metrics=compute_metrics
)

trainer.train()

model.save_pretrained("./bert_lora_sentiment_model")
tokenizer.save_pretrained("./bert_lora_sentiment_model")

swanlab.finish()