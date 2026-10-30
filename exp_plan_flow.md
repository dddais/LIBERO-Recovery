# 论文实验适配

遵守本文的的要求，进行auto exp，针对现有研究背景与问题进行实验，直到完成所有主线目标

我要睡觉了，需要我确认的部分先跳过，进行你能进行的内容。

## 基本原则（必须遵守）

- 本地代码只能参考/home/dais/workspace/FlowDAgger和/home/dais/workspace/LIBERO-Recovery的相关代码，其它仓库不允许看
- 尽量不修改现有代码库，如果需要修改，进行增量式修改，比如增加可选配置项等；
- 不允许进行git 操作本地已有的仓库，只能git clone开源仓库进行参考；
- 不允许对本地数据，结果等进行删除修改等操作，只能新增；

## 研究背景：

- dagger数据对于具身智能vla后训练至关重要，目前大多数利用方式是dagger数据聚合训练原模型checkpoint，我想探索其它的利用方式，比如flowdagger([https://arxiv.org/pdf/2607.08877)](https://arxiv.org/pdf/2607.08877)这篇工作) 这篇工作
- 目前已经复现flowdagger这篇工作 ([https://arxiv.org/pdf/2607.08877](https://arxiv.org/pdf/2607.08877)) ，目前已经拉取了开源仓库并配置了环境/home/dais/workspace/FlowDAgger/.venv-pi05，在metaworld仿真中完成了pi05的Base + FlowDAgger的12 任务主表
- 接下来我想在LIBERO-Recovery bench上完成进一步的实验验证效果



## 主线目标1

- 实验适配检查：   
  - 目标：用/home/dais/workspace/LIBERO-Recovery/datasets/plus_featured的数据对Flowdagger的pi05进行训练，并最终进行相应任务的闭环验证，基于此目标对相关代码进行了相应的适配性修改，我进行修改的原则是：
    - 1.尽可能增量式修改，不要改动原代码，比如增加并行的文件夹，增加可选配置项等等
    2.代码组织简洁高效，不要防御性编程
    3.修改完后用文档简要总结改动与使用方法
  - 请检查我上述适配修改是否合理，不合理的地方请修改相应文档和代码
  - 文档：
    - /home/dais/workspace/FlowDAgger/docs/libero_offline_usage.md；
    - /home/dais/workspace/FlowDAgger/docs/libero_offline_implementation.md



## 主线目标2

- 训练与评估
- 训练：
  - baseline1:~/workspace/model/pi05_standard_tjh (在libero多个bench的standard数据上微调后的pi05)
  - noise方法1-standard only：用/home/dais/workspace/LIBERO-Recovery/datasets/plus_featured的standard数据，按Flowdagger的方法进行训练，每个任务数据离线训一个单独的小模型
  - noise方法2-recovery only：用/home/dais/workspace/LIBERO-Recovery/datasets/plus_featured的recovery数据，按Flowdagger的方法进行训练，每个任务数据离线训一个单独的小模型
  - noise方法3-mix：用/home/dais/workspace/LIBERO-Recovery/datasets/plus_featured的recovery和standard数据按照各自50%混合，按Flowdagger的方法进行训练，每个任务数据离线训一个单独的小模型
- 评估：
  - 评估1:standard场景
  - 评估2:recovery场景
- 注意：
  - 每一个小模型用对应的任务评估
  - 注意不需要三种子，最多两种子即可
  - 注意提示词必须和训练的原始文本对齐
- 结束后用一个文档总结实验结果,要求说人话，不要防御性写作
- 结束后删除不必要的文档，代码



## 实验基础设置

增量式修改：代码修改不要影响到之前的实验运行，尽量以增量式的形式增加代码，比如加可选参数配置之类的

**文档**：使用/总结文档 放在/home/dais/workspace/FlowDAgger/docs/flow_recovery_pi05_standard_tjh

**输出**：/home/dais/workspace/FlowDAgger/outputs/flow_recovery_pi05_standard_tjh

**config** （如有必要）放在：/home/dais/workspace/FlowDAgger/config/flow_recovery_pi05_standard_tjh

**环境**：FlowDagger:/home/dais/workspace/FlowDAgger/.venv-pi05
libero:conda activate libero
starvla:conda activate starvla

**可用GPU**：0,1,2,3,4,5,6,7(最多同时使用6个，优先使用显存算力充足的卡)（建议绑定 EGL 到各自计算卡）

**vpn** : proxy_on

**模型**：~/workspace/model/pi05_standard_tjh 