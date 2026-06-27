# pipeline

## LoRA
LoRA low rank adaptation
它的核心思想是不直接修改原来的参数，而使用二个低rank，低维度的矩阵来表示原来的参数矩阵。
在训练的时候，更新这二个 low rank矩阵，快速计算。

它可以说是一种监督学习，通常训练的数据带有合适的输出。

它是实现SFT 的一种技术。SFT Supervised Fine-Tuning 是一种范式。

LoRA 在训练完可以合并到原来的模型，成为新模型。
也可以根据需要动态加载，还可以训练多个不同的LoRA，在根据实际场景加载多个不同的LoRA。

## 参数
rank 矩阵一个维度的大小。

Alpha 控制 LoRA 对原始模型的影响强度：
Alpha 越大 → LoRA 更新（BA）的贡献越大，模型适配得越“激进”，新任务/风格影响更强。
Alpha 越小 → LoRA 更新贡献越小，模型行为更接近原始预训练模型，适配更“保守”。