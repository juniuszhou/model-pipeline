# PPO Proximal Policy Optimization

## GRPO Group Relative Policy Optimization
针对同一个 Prompt，模型一次性生成 多个输出（Group）：

用 Reward Model（或规则）给每个输出打分。
计算这组分数的均值作为 baseline。
每个输出的 优势（Advantage） = （自身分数 - 组均值） / 组标准差（归一化）。
用这个相对优势更新策略模型（类似 PPO 的 clipped surrogate objective）。


## RL 
它是一种不同与监督学习，或者非监督学习的方式。它通过奖励来进行优化。
奖励信号的也可以不通过实际环境得到，比如代码是否可以编译，数学题做对了没有。
格式为一个合法的JSON

还可以使用闭源模型 LLM-as-a-Judge 来判断。


RLHF 全称是：Reinforcement Learning from Human Feedback
中文：基于人类反馈的强化学习
