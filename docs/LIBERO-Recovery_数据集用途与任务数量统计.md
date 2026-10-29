# LIBERO-Recovery 数据集用途、任务数量与 plus_featured 来源核对

核验日期：2026-10-29。本表统计单位是 **episode（示范轨迹）** 与 **frame（时间步）**，不是失败场景数、训练 batch 数或相机图片数；双相机不会把时间帧数乘二。

官方 Expert 统计取自 ModelScope `ataier/LIBERO_Recovery_Expert` 的固定版本 **2804e989adfb49cdde9b5c04bc7e0fb9f4fbfc73**。本地统计依据 `datasets/plus_featured/data_pool.json`、`tasks.csv`、`schema.json` 和解压后的数组。本次只下载官方小型元数据，没有下载全部视频或运行训练。

## 1. 这么多数据集分别做什么，训练应选哪个

| 数据集/资源 | 主要内容 | 用途 | 是否适合直接作为成功动作示范训练 |
|---|---|---|---|
| `LIBERO_Recovery_Expert` | 人工从失败状态恢复任务的 LeRobot 示范，3,184 episodes | 恢复策略的监督训练、微调 | **是，训练恢复能力的主要官方数据**；先检查单帧记录、任务映射与质量 |
| 原始 LIBERO demonstrations | 从正常初始状态完成任务的专家示范 | 基础动作策略训练，或与恢复示范混合 | **是**；恢复微调时建议保留正常示范以检查是否损害原任务能力 |
| `LIBERO_10_LL` | LIBERO-10 执行中失败前的历史片段 | 时间上下文、失败前状态分析 | 可以作为特定历史输入研究的素材；不等同于完整成功恢复轨迹 |
| `LIBERO_Recovery_FailureScenes` | 当前发布的 4,803 个 BDDL/状态文件对及标注、索引 | 从失败状态启动仿真，评测恢复能力 | **不是直接的动作标签数据集**；可作为新一轮采集或 RL 的起点 |
| `LIBERO_Recovery_Assets` | 精选场景资源，`scenes.tar.gz` 有 880 个 BDDL；包内另有更大快照归档 | 场景初始化、精选场景评测 | 同样不是专家动作示范；不同场景包不要不加区分地合并 |
| `LIBERO_Recovery_Rollouts` | 策略原始执行轨迹，README 报告 1,446 条、约 80 GB | 失败定位、标注、场景构建溯源 | 包含模型执行失败，不应不加标签地当作成功专家做行为克隆 |
| 本地 `plus_featured` | 10 个指定任务：500 条正常示范 + 318 条 Expert 恢复示范 | 小规模、任务匹配的正常/恢复混合训练 | **是，适合先跑这 10 个任务的训练闭环**；使用 NumPy 缓存读取适配，不能直接按 LeRobot 目录读取 |

**选择建议：**如果要训练广覆盖的恢复能力，使用官方 Expert，并与原始 LIBERO 正常示范混合；如果先验证本地这 10 个任务的恢复微调方案，优先使用已经配齐正常与恢复数据的 `plus_featured`。恢复数据多不意味着正常执行一定更好，训练后应分别评测正常初态和失败初态。

`FailureScenes`、`Assets` 的数量是“可启动评测的场景数”，Expert 的数量是“专家示范条数”；同一场景可能有多条示范，二者不能直接比较。

## 2. 官方 Expert 的任务数量：先澄清统计口径

| 项目 | 实测值 | 解释 |
|---|---:|---|
| `info.json.total_tasks` / `tasks.jsonl` 行数 | 110 | 110 个 task index 条目 |
| 原始任务指令字符串去重 | **78** | 110 条中有重复字符串，不能把重复条目的聚合数量重复相加 |
| 全部轨迹 | **3,184** | 和 `info.json` 一致 |
| 全部时间帧 | **625,731** | 和 `info.json` 一致 |
| 仅 1 帧的轨迹 | **373** | 占约 11.7%；不能提供有效的多步恢复过程 |
| 长度大于 1 帧的轨迹 | **2,811** | 是一个基础过滤后的候选训练集，不等于已逐条认证成功 |
| 多帧轨迹的时间帧 | **625,358** | 过滤掉 373 个单帧记录后 |
| 官方 split | `train: 0:3184` | 元信息没有另外给出独立验证/测试 split |

下表按 **完全一致的原始指令字符串** 合并，保留对应的全部 task index。没有把 `pick ...` 与 `pick up ...`、`rack` 与 `wine rack` 等近义文字自动合并。78 是唯一指令字符串数，不能仅凭这些元数据断言等于 78 个不同物理场景或语义任务。

`episodes.jsonl` 给出的是任务文字和长度，未给出权威的失败场景 ID/suite 映射。表中的数量是同一文字对应的轨迹总数，**不是把总数分别赋给多个重复 task index**。中文简称仅供阅读；精确指令见表内英文与 CSV。

完整机器可读表：[official_expert_task_counts.csv](dataset_statistics/official_expert_task_counts.csv)。

### 2.1 全部 78 条唯一指令的统计表

| 官方 task index（同文合并） | 原始任务指令 | 全部轨迹 | 多帧轨迹 | 单帧轨迹 | 总帧数 |
|---|---|---:|---:|---:|---:|
| 0,38,56 | put the black bowl in the top drawer of the cabinet | 35 | 35 | 0 | 13,229 |
| 1,39 | put the butter at the back in the top drawer of the cabinet and close it | 20 | 20 | 0 | 12,109 |
| 2,40 | put the butter at the front in the top drawer of the cabinet and close it | 20 | 20 | 0 | 10,647 |
| 3,44 | open the top drawer of the cabinet and put the bowl in it | 25 | 25 | 0 | 8,492 |
| 4,47 | put the black bowl at the front on the plate | 40 | 40 | 0 | 5,993 |
| 5 | put the middle black bowl on top of the cabinet | 15 | 15 | 0 | 2,455 |
| 6,49 | stack the black bowl at the front on the black bowl in the middle | 20 | 20 | 0 | 4,152 |
| 7,50 | stack the black bowl in the middle on the black bowl at the front | 20 | 20 | 0 | 5,348 |
| 8,52 | put the moka pot on the stove | 25 | 25 | 0 | 7,388 |
| 9,64 | turn on the stove and put the frying pan on it | 10 | 10 | 0 | 1,667 |
| 10,53 | close the bottom drawer of the cabinet and open the top drawer | 10 | 10 | 0 | 1,542 |
| 11,54 | put the black bowl on top of the cabinet | 10 | 10 | 0 | 1,823 |
| 12 | put the wine bottle in the bottom drawer of the cabinet | 5 | 5 | 0 | 456 |
| 13,55 | put the wine bottle on the wine rack | 10 | 10 | 0 | 4,292 |
| 14,57 | put the ketchup in the top drawer of the cabinet | 25 | 25 | 0 | 9,272 |
| 15,59 | put the yellow and white mug to the front of the white mug | 10 | 10 | 0 | 1,408 |
| 16,61 | put the white bowl to the right of the plate | 10 | 10 | 0 | 2,247 |
| 17 | turn off the stove | 5 | 5 | 0 | 733 |
| 18 | put the frying pan on the cabinet shelf | 5 | 5 | 0 | 705 |
| 19,65 | pick up the alphabet soup and put it in the basket | 15 | 15 | 0 | 3,087 |
| 20 | pick up the ketchup and put it in the basket | 5 | 5 | 0 | 1,267 |
| 21,66,69 | pick up the tomato sauce and put it in the basket | 20 | 20 | 0 | 3,310 |
| 22,67 | pick up the butter and put it in the basket | 10 | 10 | 0 | 1,175 |
| 23,68 | pick up the milk and put it in the basket | 15 | 15 | 0 | 2,514 |
| 24,70 | pick up the alphabet soup and put it in the tray | 10 | 10 | 0 | 2,041 |
| 25,71 | pick up the ketchup and put it in the tray | 10 | 10 | 0 | 2,429 |
| 26 | stack the left bowl on the right bowl and place them in the tray | 15 | 15 | 0 | 4,359 |
| 27,74 | stack the right bowl on the left bowl and place them in the tray | 20 | 19 | 1 | 6,096 |
| 28,75 | put the red mug on the left plate | 10 | 10 | 0 | 2,619 |
| 29,76 | put the white mug on the left plate | 15 | 15 | 0 | 2,068 |
| 30,80 | put the white mug on the plate | 20 | 20 | 0 | 2,545 |
| 31,81,83 | pick up the book and place it in the front compartment of the caddy | 15 | 15 | 0 | 5,218 |
| 32 | pick up the book and place it in the left compartment of the caddy | 5 | 5 | 0 | 488 |
| 33,84 | pick up the book and place it in the right compartment of the caddy | 9 | 9 | 0 | 3,852 |
| 34 | pick up the book and place it in the back compartment of the caddy | 5 | 5 | 0 | 2,224 |
| 35,86 | pick up the white mug and place it to the right compartment of the caddy | 10 | 10 | 0 | 2,780 |
| 36,88 | pick up the book on the right and place it under the cabinet shelf | 10 | 10 | 0 | 3,234 |
| 37 | close the top drawer of the cabinet and put the black bowl on top of it | 16 | 16 | 0 | 5,365 |
| 41 | put the chocolate pudding in the top drawer of the cabinet and close it | 5 | 5 | 0 | 1,594 |
| 42 | open the bottom drawer of the cabinet | 20 | 20 | 0 | 3,428 |
| 43 | open the top drawer of the cabinet | 5 | 5 | 0 | 673 |
| 45 | put the black bowl on the plate | 5 | 5 | 0 | 687 |
| 46 | put the black bowl at the back on the plate | 10 | 10 | 0 | 1,766 |
| 48 | put the black bowl in the middle on the plate | 5 | 5 | 0 | 875 |
| 51 | put the frying pan on the stove | 5 | 5 | 0 | 1,072 |
| 58 | close the microwave | 5 | 5 | 0 | 1,138 |
| 60 | put the white bowl on the plate | 5 | 5 | 0 | 540 |
| 62 | put the right moka pot on the stove | 5 | 5 | 0 | 1,174 |
| 63 | put the frying pan on top of the cabinet | 5 | 5 | 0 | 989 |
| 72 | pick up the tomato sauce and put it in the tray | 5 | 5 | 0 | 1,059 |
| 73 | pick up the salad dressing and put it in the tray | 5 | 5 | 0 | 687 |
| 77 | put the yellow and white mug on the right plate | 5 | 5 | 0 | 750 |
| 78 | put the chocolate pudding to the left of the plate | 5 | 5 | 0 | 470 |
| 79 | put the red mug on the plate | 5 | 5 | 0 | 1,516 |
| 82 | pick up the yellow and white mug and place it to the right of the caddy | 5 | 5 | 0 | 986 |
| 85 | pick up the red mug and place it to the right compartment of the caddy | 5 | 5 | 0 | 560 |
| 87 | pick up the book on the right and place it on the cabinet shelf | 5 | 5 | 0 | 1,019 |
| 89 | open the middle layer of the drawer | 21 | 21 | 0 | 2,634 |
| 90 | open the top layer of the drawer and put the bowl inside | 36 | 36 | 0 | 9,442 |
| 91 | push the plate to the front of the stove | 21 | 21 | 0 | 3,589 |
| 92 | put the bowl on the plate | 5 | 5 | 0 | 381 |
| 93 | put the bowl on the top of the drawer | 5 | 5 | 0 | 965 |
| 94 | put the cream cheese on the bowl | 23 | 23 | 0 | 3,006 |
| 95 | put the wine bottle on the rack | 33 | 33 | 0 | 8,376 |
| 96 | put the wine bottle on the top of the drawer | 35 | 35 | 0 | 9,520 |
| 97 | pick the alphabet soup and place it in the basket | 269 | 267 | 2 | 70,583 |
| 98 | pick the bbq sauce and place it in the basket | 203 | 201 | 2 | 45,456 |
| 99 | pick the butter and place it in the basket | 62 | 61 | 1 | 14,723 |
| 100 | pick the chocolate pudding and place it in the basket | 225 | 223 | 2 | 35,988 |
| 101 | pick the cream cheese and place it in the basket | 91 | 90 | 1 | 18,069 |
| 102 | pick the tomato sauce and place it in the basket | 300 | 300 | 0 | 48,221 |
| 103 | pick the akita black bowl between the plate and the ramekin and place it on the plate | 251 | 1 | 250 | 428 |
| 104 | pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate | 308 | 198 | 110 | 55,577 |
| 105 | pick the akita black bowl next to the cookies box and place it on the plate | 103 | 103 | 0 | 20,407 |
| 106 | pick the akita black bowl next to the plate and place it on the plate | 26 | 26 | 0 | 3,987 |
| 107 | pick the akita black bowl on the cookies box and place it on the plate | 52 | 52 | 0 | 8,284 |
| 108 | pick the akita black bowl on the stove and place it on the plate | 182 | 180 | 2 | 31,798 |
| 109 | pick the akita black bowl on the wooden cabinet and place it on the plate | 263 | 261 | 2 | 62,685 |
| **合计** | **78 条唯一指令** | **3,184** | **2,811** | **373** | **625,731** |

其中两个任务的原始数量尤其容易误导：

- `task_index=103`：盘子与 ramekin 之间的黑碗放到盘子上，251 条中 **250 条只有一帧**，多帧仅 1 条。
- `task_index=104`：木柜上层的黑碗放到盘子上，308 条中 **110 条只有一帧**，多帧为 198 条。

过滤这些记录几乎不改变总帧数，却会显著改变按 episode 均匀采样时的训练分布。多帧条件只是最基本的长度过滤，不能替代成功终止、动作连续性和视频质量检查。

## 3. plus_featured：每个任务有多少条数据

**总计 10 个任务、818 条轨迹、159,744 个时间帧**，其中正常示范 500 条/81,383 帧，恢复示范 318 条/78,361 帧。每个任务恰好有 50 条正常示范；恢复数量不均衡。

完整表：[plus_featured_task_counts.csv](dataset_statistics/plus_featured_task_counts.csv)。以下任务编号是本地的 `task_index=0..9`，不是官方 Expert 的 task index。

| 本地编号 | Suite | 任务 | 正常轨迹 | 恢复轨迹 | 总轨迹 | 正常帧数 | 恢复帧数 |
|---:|---|---|---:|---:|---:|---:|---:|
| 0 | libero_90 | 前方黑碗放到盘子上 | 50 | 40 | 90 | 6,893 | 5,993 |
| 1 | libero_90 | 打开上层抽屉并放入碗 | 50 | 25 | 75 | 9,994 | 8,492 |
| 2 | libero_90 | 番茄酱放入上层抽屉 | 50 | 25 | 75 | 10,512 | 9,272 |
| 3 | libero_90 | 打开底层抽屉 | 50 | 20 | 70 | 7,156 | 3,428 |
| 4 | libero_90 | 前方黑碗叠到中间黑碗上 | 50 | 20 | 70 | 6,415 | 4,152 |
| 5 | libero_90 | 摩卡壶放到炉灶上 | 50 | 25 | 75 | 7,553 | 7,388 |
| 6 | libero_90 | 白色马克杯放到盘子上 | 50 | 20 | 70 | 7,885 | 2,545 |
| 7 | libero_90 | 前方黄油放入上层抽屉并关闭 | 50 | 20 | 70 | 9,008 | 10,647 |
| 8 | libero_goal | 葡萄酒瓶放到酒架上 | 50 | 33 | 83 | 8,801 | 8,376 |
| 9 | libero_object | 奶油奶酪放入篮子 | 50 | 90 | 140 | 7,166 | 18,068 |
| **合计** | **10 个任务** | | **500** | **318** | **818** | **81,383** | **78,361** |

按 suite 汇总：

| Suite | 任务数 | 正常轨迹 | 恢复轨迹 | 总轨迹 |
|---|---:|---:|---:|---:|
| libero_90 | 8 | 400 | 195 | 595 |
| libero_goal | 1 | 50 | 33 | 83 |
| libero_object | 1 | 50 | 90 | 140 |
| 合计 | 10 | 500 | 318 | 818 |

这个子集不包含 Spatial 或 libero_10 任务，不能将其训练结果概括为覆盖了全部 LIBERO-Recover。

## 4. plus_featured 和官方 Expert 到底是什么关系

### 4.1 恢复部分是 Expert 的筛选和格式转换

按 `data_pool.json` 保存的原始 episode ID、原始指令和长度，逐条对照固定版本官方元数据，**318 条全部匹配**。进一步检查集合相等性得到：本地恢复部分恰好等于这 10 条原始指令对应的所有 `length>1` 官方轨迹，没有额外采样截断。

| 本地任务 | 官方 Expert 原始轨迹 | 排除单帧 | 本地保留恢复轨迹 |
|---|---:|---:|---:|
| 前方黑碗放到盘子上 | 40 | 0 | 40 |
| 打开上层抽屉并放入碗 | 25 | 0 | 25 |
| 番茄酱放入上层抽屉 | 25 | 0 | 25 |
| 打开底层抽屉 | 20 | 0 | 20 |
| 前方黑碗叠到中间黑碗上 | 20 | 0 | 20 |
| 摩卡壶放到炉灶上 | 25 | 0 | 25 |
| 白色马克杯放到盘子上 | 20 | 0 | 20 |
| 前方黄油放入上层抽屉并关闭 | 20 | 0 | 20 |
| 葡萄酒瓶放到酒架上 | 33 | 0 | 33 |
| 奶油奶酪放入篮子 | 91 | 1 | 90 |
| **合计** | **319** | **1** | **318** |

唯一排除项是官方 `episode_index=1672`，原始指令 `pick the cream cheese and place it in the basket`，长度为 1。本地奶油奶酪任务因此是 90 条恢复示范，而官方这一指令是 91 条。

318 条约占官方全部轨迹的 **9.99%**，占官方多帧轨迹的 **11.31%**；对应 78,361 帧约占官方总帧数的 **12.52%**。

逐 episode 的对应关系见 [plus_featured_expert_episode_mapping.csv](dataset_statistics/plus_featured_expert_episode_mapping.csv)。这种关系已经由元数据独立验证；本次没有把官方全部视频下载下来逐像素比对，不能将“来源记录一致”扩大为“所有上游字节都已独立核验”。

### 4.2 正常示范来自原始 LIBERO，不来自 Expert

500 条 Standard 的来源记录指向 `yifengzhu-hf/LIBERO-datasets`，固定 revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`。其中 400 条属于所选 8 个 libero_90 任务，另外 100 条为 Goal 葡萄酒瓶和 Object 奶油奶酪任务的原始 HDF5 示范，每任务 50 条。

打包说明记录：后两个任务的 100 条从原始 HDF5 重建，其中 81 条对应旧 LeRobot cache，19 条是从原始文件补回的不同示范。**“新增 19 条”是相对于旧缓存，不能理解成新采集的恢复轨迹。** 打包者还记录这 100 条做过源场景 XML 的起终态成功检查；本次没有重新运行这项仿真检查。

正常示范原始共 81,573 帧，缓存删除 190 个静止动作帧后为 81,383 帧；恢复部分保留 78,361 帧，`noops_removed` 为 0。

### 4.3 两者的训练格式不同

| 项目 | 官方 Expert | plus_featured |
|---|---|---|
| 存储 | LeRobot v2.1：Parquet + 双相机 MP4 + meta | 每 episode 三个 `.npy`，按任务用 `.tar.zst` 打包 |
| 图像 | 两路视频 | `images[T,2,256,256,3]`，uint8 RGB，顺序 agentview/wrist |
| 状态 | `observation.state[T,8]` | `state[T,8]`，float32：末端位置3、轴角3、两手指关节2 |
| 动作 | `action[T,7]` | `actions[T,7]`，float32：6 维 OSC_POSE 控制量 + 夹爪 |
| 夹爪约定 | 包的转换来源说明为 `0=close,1=open` | 已用 `1-2*a` 转成原生 LIBERO：`-1=open,+1=close` |
| 指令 | 原始采集任务文字，有重复及表达差异 | 统一为官方任务语言，原文保留在 `raw_prompt` |
| 图像方向 | 取决于原始来源 | 包说明已对齐训练/运行方向；不要再次无条件旋转 |

这里的“统一指令”不等于数值归一化。NumPy 缓存保存的是未经训练统计归一化的动作与状态；接入模型时仍需按该模型的数据契约处理。恢复夹爪已经转换，不能第二次应用 `1-2*a`；末端位姿增量也不能误当作关节位置或绝对末端目标。

### 4.4 训练时避免重复计算与数据泄漏

- 若训练集已经包含完整 Expert，再直接加入 `plus_featured` 的恢复部分，会重复采样这 318 条。只有明确设计了重加权时才这样做。
- `plus_featured` 不能作为“独立于 Expert 的恢复测试集”，因为恢复 episode 本身来自 Expert。
- 一个任务多帧不等于它附有失败前历史；`plus_featured` 保留多帧恢复过程，并不是 `LIBERO_10_LL` 的历史拼接包。
- 当前任务分配依据是唯一匹配的指令，而非作者提供的失败场景映射。不能仅凭任务名将这些轨迹与 4,803 个场景精确一一配对。
- 数据包没有提供独立的训练/验证划分。应按原始 episode、可用的场景来源分组划分，避免随机按帧拆分相邻片段；同一任务多条示范也可能来自同一失败场景。
- 本地子集是训练用 NPY 缓存，不包含足以替代 BDDL+完整仿真状态的重放初态；闭环评测仍需要另选评测场景。

## 5. 本次核验做了什么

1. 从官方固定 revision 获取 `info.json`、`tasks.jsonl`、`episodes.jsonl`，三个文件的 SHA256 均与官方文件树记录相符；原始元数据保存在 [official_expert_meta/](dataset_statistics/official_expert_meta/)。
2. 从 episode 元信息重新汇总 78 条唯一指令、3,184 条轨迹、625,731 帧；统计 373 条单帧与 2,811 条多帧。
3. 从本地逐 episode 记录重新汇总 10 个任务，并与 `tasks.csv`、manifest 的总量交叉核对。
4. 检查本地全部 **2,454 个 NumPy 数组**的存在性及形状；全部 **1,636 个状态/动作数组**均通过打包清单 SHA256 和有限数值检查；两类数据的夹爪取值范围均为 [-1,1]。本次没有为所有图像数组重新计算完整内容哈希。
5. 按 episode ID、任务文字和长度逐条比对 318 条恢复记录，并确认它们是对应官方任务的完整多帧子集。

核验汇总：[audit_summary.json](dataset_statistics/audit_summary.json)。没有修改数据包、训练代码或原有实验计划。

## 6. 来源

- [官方 Expert 数据集](https://www.modelscope.cn/datasets/ataier/LIBERO_Recovery_Expert)：固定 revision 如上。
- [Expert info.json](https://www.modelscope.cn/api/v1/datasets/ataier/LIBERO_Recovery_Expert/repo?Revision=2804e989&FilePath=meta/info.json)、[tasks.jsonl](https://www.modelscope.cn/api/v1/datasets/ataier/LIBERO_Recovery_Expert/repo?Revision=2804e989&FilePath=meta/tasks.jsonl)、[episodes.jsonl](https://www.modelscope.cn/api/v1/datasets/ataier/LIBERO_Recovery_Expert/repo?Revision=2804e989&FilePath=meta/episodes.jsonl)：本次计数的直接依据。
- [本地 plus_featured README](../datasets/plus_featured/README.md)、[tasks.csv](../datasets/plus_featured/tasks.csv)、[data_pool.json](../datasets/plus_featured/data_pool.json)、[schema.json](../datasets/plus_featured/schema.json)：来源、格式与任务对应关系。
- [原始 LIBERO 示范](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)：正常示范的上游来源。
- [仓库 README](../README.md)、[论文与代码使用指南](LIBERO-Recovery_论文解读与代码使用指南.md)：其他发布数据集用途及评测代码说明。
