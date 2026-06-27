1. SFT Supervised Fine-Tuning（有监督微调 / 指令微调）强烈推荐 / 几乎必须★★★★★让模型学会听指令、按格式输出

2. Alignment 对齐阶段（偏好优化）推荐，但非绝对必须★★★★☆让模型符合人类偏好、安全、诚实

3. 量化（Quantization）4bit / 8bit 量化部署时必须★★★★降低显存和推理成本



4. 蒸馏（Distillation）知识蒸馏可选★★用于做小模型

5. 继续预训练（Continued Pre-training）领域继续预训练



PEFT 是 Parameter-Efficient Fine-Tuning
