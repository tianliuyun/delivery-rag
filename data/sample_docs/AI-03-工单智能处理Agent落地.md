## 工单智能处理 Agent 落地指南

### 1. 总体流程设计

工单智能处理 Agent 按四阶段流水线运行，各阶段通过内部消息队列（NATS，默认端口 `4222`）串联。

1. **工单分类初筛**：接入工单后，调用轻量分类模型（如 `bge-small-zh-v1.5` + 逻辑回归），输出一级分类（网络/计算/存储/数据库/安全）及置信度。
2. **多 Worker 并行诊断**：按分类结果并发调度 3~5 个诊断 Worker（Python 进程，监听 `tcp://127.0.0.1:5557`），每个 Worker 执行独立检查脚本，超时阈值 `15s`。
3. **技能匹配分派**：汇总诊断结果，匹配技能标签（如 `k8s-cni`、`ceph-osd`），按权重 `0.6*技能匹配度 + 0.4*当前负载` 选择工程师，通过 Webhook 推送到工单系统。
4. **低置信人工复核**：当分类置信度 `< 0.75` 或诊断结果冲突率 `> 30%` 时，自动转人工复核队列（Redis List `review_queue`，端口 `6379`）。

### 2. 关键配置与命令

#### 2.1 分类初筛服务启动
```bash
python -m classifier.server --port 8001 --model-path /models/cls_v3 \
  --confidence-threshold 0.75 --batch-size 16
```
若置信度低于阈值，直接写入 `review_queue`，跳过后续阶段。

#### 2.2 多 Worker 并行诊断
```bash
# 启动 4 个 Worker，绑定不同端口
for i in {0..3}; do
  python -m worker.diagnose --worker-id $i --port $((5557+i)) \
    --max-concurrency 3 --timeout 15 &
done
```
诊断脚本示例：`check_network.sh` 使用 `ping -c 3 -W 2` 和 `traceroute -m 10`。

#### 2.3 技能匹配分派
分派服务读取 `skills.yaml`：
```yaml
skills:
  - name: k8s-cni
    keywords: ["cni", "calico", "flannel"]
    weight: 0.8
```
匹配命令：
```bash
python -m dispatcher.match --ticket-id T20250401-001 \
  --diagnosis-json /tmp/diag.json --top-k 3
```

### 3. 评测集构建（可复现样例）

构建 200 条领域知识样例，覆盖 5 个一级分类，每类 40 条。样例格式：
```json
{
  "ticket_id": "eval-001",
  "text": "Pod 无法启动，报错 failed to allocate IP",
  "ground_truth": "网络",
  "diagnosis_expected": ["cni-plugin-down", "ipam-exhausted"],
  "confidence_expected": 0.92
}
```
**可复现要求**：每条样例附带 `seed` 和 `mock_data`，使用 `pytest` 固定随机种子：
```bash
pytest tests/eval --seed 42 --num-samples 200
```
样例来源：历史工单脱敏 + 合成增强（`nlpaug` 同义词替换，比例 1:1）。

### 4. A/B 对比验证方法

将流量按 `ticket_id` 哈希取模分为 A（对照组，纯人工）和 B（实验组，Agent 流水线），比例 `50%:50%`。

**监控指标**：
- 平均处理时长（MTTR）
- 首次分派准确率（FAR）
- 人工复核率（HRR）
- 工单重开率（RR）

**验证命令**：
```bash
python -m abtest.analyze --start 2025-04-01 --end 2025-04-14 \
  --group-a-file /logs/group_a.csv --group-b-file /logs/group_b.csv \
  --metrics mttr,far,hrr,rr --alpha 0.05
```
若 B 组 MTTR 降低 `> 20%` 且 FAR 提升 `> 15%`，且 p-value `< 0.05`，则判定 Agent 有效。否则回滚至纯人工流程，并检查分类阈值或 Worker 超时参数。