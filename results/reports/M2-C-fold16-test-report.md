# M2-C Sector-GCN 真实数据测试报告

## 1. 测试范围

- 数据集：`/Volumes/Samsung_T5/ready_for_use.zip/ready_for_use/ready_for_use`
- 数据规模：36 GB，2000-2025，17 个 walk-forward folds，每日 500 个节点
- 测试 fold：16
- 训练期：2016-2023（2,012 个交易日，1,001,941 个有效监督节点）
- 验证期：2024（252 个交易日，125,241 个有效监督节点）
- 测试期：2025-01-02 至 2025-12-31（250 个交易日，124,178 个有效监督节点）
- 图结构：同 `sector_index` 节点构成完整 clique，加入一次自环
- 测试性质：真实数据单-fold短调参测试，不是完整 17-fold 最终实验

## 2. 训练配置

- 优化器：Adam
- 每批日期数：32
- 最大 epoch：20
- Early-stopping patience：5
- L2：0.0001
- 学习率：0.003
- 候选隐藏维度：16、32
- 选择指标：验证集 daily-equal cross-entropy
- 选择后从头在 2016-2024 上 refit，再生成一次 2025 测试概率

复现实验命令：

```bash
PYTHONPATH=src /opt/homebrew/anaconda3/bin/python script/fit_m2_gcn.py \
  --input-dir /Volumes/Samsung_T5/ready_for_use.zip/ready_for_use/ready_for_use \
  --output-dir results/models/M2-C-fold16-test \
  --config configs/testing/M2-C-fold16.json \
  --fold 16
```

候选结果：

| 候选 | 最佳 epoch | 最佳验证 CE | 实际 epochs | 训练耗时 |
|---|---:|---:|---:|---:|
| h16 | 6 | 1.586551 | 11 | 24.17 秒 |
| h32 | 6 | **1.586373** | 11 | 38.21 秒 |

最终选择 `h32_lr0p003_l2_1em4`，最佳 epoch 为 6。

## 3. 验证与测试指标

| 数据段 | Cross-entropy ↓ | Accuracy ↑ | Rank IC ↑ | Pearson IC ↑ |
|---|---:|---:|---:|---:|
| 2024 验证 | 1.586373 | 23.8812% | 0.011681 | 0.005247 |
| 2025 测试 | 1.583109 | 24.0766% | 0.003467 | 0.003002 |

均匀五分类基准为 CE=`log(5)=1.609438`、Accuracy=20%。因此 M2-C 在真实测试集上优于均匀分类基准，但 2025 日度 Rank IC 的 t 值仅为 0.409，未显示统计显著性。

## 4. 与同一 2025 fold 基线比较

| 模型 | Test CE ↓ | Accuracy ↑ | Rank IC ↑ | Pearson IC ↑ |
|---|---:|---:|---:|---:|
| M2-C GCN | 1.583109 | 24.0766% | 0.003467 | 0.003002 |
| M0 Logistic | **1.564837** | 25.8138% | 0.006270 | 0.004022 |
| M0-Plus XGBoost | **1.558164** | **26.0562%** | **0.015125** | **0.017338** |

相对 M0，M2-C：

- CE 高 0.018272（更差）
- Accuracy 低 1.737 个百分点
- Rank IC 低 0.002804
- Pearson IC 低 0.001020

注意：M0/M0-Plus 使用其已有正式配置，M2-C 本次只测试两个候选、最多 20 epochs，因此比较适合用于诊断，不应视为最终模型排名。

## 5. 2025 多空组合诊断

组合规则为按预测期望类别分数排序，20% 多头、20% 空头、等权、美元中性、每日再平衡。

| 模型 | 0 bps Sharpe | 0 bps 日均收益 | 10 bps Sharpe | 10 bps 日均净收益 | 日均换手 |
|---|---:|---:|---:|---:|---:|
| M2-C GCN | -0.245 | -0.915 bps | -1.375 | -5.141 bps | 0.423 |
| M0 Logistic | -0.468 | -1.516 bps | **-1.371** | **-4.444 bps** | **0.293** |
| M0-Plus XGBoost | -0.007 | **-0.026 bps** | -1.684 | -6.227 bps | 0.620 |

M2-C 未计成本时比 M0 少亏 0.601 bps/日，但换手高约 44%，使 10 bps 成本后净收益比 M0 差 0.696 bps/日。2025 年三个模型的该多空组合均未实现正收益。

## 6. 效率与完整性

| 指标 | M2-C | M0 | M0-Plus |
|---|---:|---:|---:|
| 参数量 | 22,533 | 3,495 | 不适用 |
| 单 fold 总耗时 | 93.89 秒 | 35.23 秒 | 126.93 秒 |
| 2025 推理耗时 | 0.235 秒 | 0.011 秒 | 1.622 秒 |

- 输出概率形状：`[250, 500, 5]`
- 249 个日期可评估，最后一个日期没有下一交易日收益
- 概率、checkpoint 和 prediction manifest 哈希校验通过
- 再次执行同一命令时成功复用已完成 fold，没有重复训练或测试推理
- 结果目录大小：17 MB

## 7. 结论

本次真实数据测试证明 M2-C 训练、验证选模、refit、测试与 OOS 输出流程可正常运行，数值稳定且优于均匀分类基准。但该严格 sector-clique GCN 在 2025 的 CE、Accuracy 和 IC 均未超过 M0/M0-Plus，Rank IC 也不显著。

主要结构性原因是完整 sector clique 会精确平均组内节点，使同 sector 股票获得相同预测。本次 M2-C 日均只有约 184 个不同分数，而 M0 接近 499 个，显著限制了 sector 内选股能力。下一步更值得测试的是单独命名的 residual GCN，或使用稀疏 correlation/lead-lag 图的 M2-A/M2-B。
