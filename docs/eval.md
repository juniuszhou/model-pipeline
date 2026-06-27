# Model evaluation

## MMLU（Massive Multitask Language Understanding）

## HumanEval ability of coding

## project
https://github.com/confident-ai/deepeval




100 个样本  60个对  40个错
你找了50个  35 个对  15个错

TP	True Positive（预测正确的正样本）正样本预测正确   35
FP	False Positive（预测错的正样本） 正样本预测错误   15
FN	False Negative（漏掉的正样本）   正样本没有被找到  25
TN	True Negative（预测正确的负样本） 负样本没有选出来  25

Precision = TP / （FP +TP​）  35 / （50）70%  准不准
Recall = TP / （TP + FN）    35 / （60）     全不全

F1​ = 2⋅ Precision⋅ Recall​ / （Precision+Recall）

