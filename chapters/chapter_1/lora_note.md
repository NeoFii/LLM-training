## 通过Peft封装的LoRA模块微调BERT

借助PEFT进行LoRA微调的流程步骤如下：

1. 设定好lora_config，包含TaskType
2. 导入模型
3. 将模型和lora_config通过`get_peft_model()`合并为一个PeftModel类实例
4. 训练时将合并后的PeftModel实例传递给Trainer的model，设定好训练参数即可训练

第三步完成后可以使用`print_trainable_parameters()`方法验证是否冻结了预训练模型权重。
假设有`model = get_peft_model(model, lora_config)`，则`print(model.print_trainable_parameters())`，出现类似:

trainable params: 300,000 || all params: 110,000,000 || trainable%: 0.27%的信息。

---
### 不同参数微调下BERT的表现

详细的训练结果：https://swanlab.cn/@reed/bert_lora/runs

结果发现：

1. LoRA相比全量微调，并没有出现跨epoch train_loss骤降的现象，整个训练过程中未出现阶梯状loss
2. 暂时没有看到训练对lr的依赖情况，对比lr线性衰减和前10%warmup后cosine退火，最终各项指标几乎一样
3. 增大r后eval降得快，eval_acc升的快，f1值略有升高，recall升高2个点，但precision下降1.2个点
4. 增加attention层的输出映射矩阵作为lora训练目标后发现整体各项性能相比target_modules=["query", "key", "value"]更高，说明对于我们实验的bert微调文本分类任务，适当增大训练的参数能够带来一定的性能提升
5. 继续增大r后收益不再明显


整个训练过程中没有发生过拟合，模型整体性能良好。但是得分上不如全量微调。

但是增大r和添加更多的target_module本质上都是增大了可训练的参数，结果发现前者的提升效果并不明显。我的理解是，选定modules时增大r相当于加深了module层的学习深度；固定r增加modules相当于深度一定扩宽了广度。LoRA的核心假设是更新权重 $\Delta W$具有极低的"本征秩"，r变大并不会改变"本征秩"的大小，所以增大r收效甚微。而output映射矩阵负责将QKV计算的相关信息进行线性组合，融合各部分的信息，后续的FFN层负责处理信息。将这些线性层一起训练后能够有效的提升下游任务上的表现。

可以做的更细粒度，从attention模块开始，一次只训练一个子块，逐渐增加能够找到最佳的target_modules。