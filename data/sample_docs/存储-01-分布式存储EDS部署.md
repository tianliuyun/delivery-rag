## 分布式存储（EDS类）部署指南

### 1. 存储节点规划

#### 1.1 硬件配置基线
EDS 采用 SSD 缓存 + HDD 容量的混合架构。单节点推荐配置如下：

| 组件 | 规格 | 数量 | 说明 |
|------|------|------|------|
| CPU | Intel Xeon Silver 4314 或以上 | 2 | 16C/32T 起 |
| 内存 | 128GB DDR4 ECC | 8 | 每 1TB 裸容量配 4GB 内存 |
| SSD 缓存盘 | NVMe 1.92TB 企业级 | 2 | 做 RAID1，用于元数据与写缓存 |
| HDD 容量盘 | 16TB SATA 7.2K | 10 | 单节点裸容量 160TB |
| 网卡 | 双口 10GbE SFP+ | 1 | 存储网络专用，做 bond |
| 系统盘 | 480GB SATA SSD | 2 | RAID1 安装 OS |

#### 1.2 节点角色划分
- **管理节点**：3 节点，部署 EDS 管理服务与元数据服务，不参与数据存储。
- **存储节点**：≥3 节点，每节点 SSD:HDD 容量比控制在 1:10 以内。
- **客户端节点**：通过 `eds-client` 挂载，无需本地盘。

#### 1.3 容量规划公式
可用容量 = 裸容量 × 0.85（RAID/EC 开销）× 0.9（预留水位）。例如 3 节点 × 160TB = 480TB 裸容量，三副本后可用约 122TB。

### 2. 网络要求

#### 2.1 网络分区
- **前端业务网**：10GbE，承载客户端读写，MTU 1500。
- **后端存储网**：10GbE 或 25GbE，承载副本复制与数据重建，MTU 9000（Jumbo Frame）。
- **管理网**：1GbE，承载监控与 API。

#### 2.2 端口清单
| 端口 | 协议 | 用途 |
|------|------|------|
| 22 | TCP | SSH 运维 |
| 80/443 | TCP | Web 管理控制台 |
| 6789 | TCP | 监控服务（Ceph Mon 类） |
| 6800-7300 | TCP | OSD 数据服务 |
| 2049 | TCP | NFS 导出 |
| 3260 | TCP | iSCSI 目标 |
| 8443 | TCP | REST API |

#### 2.3 网络配置命令（每存储节点）
```bash
# 绑定双口万兆
nmcli con add type bond con-name bond0 ifname bond0 mode 802.3ad
nmcli con add type ethernet con-name bond0-slave1 ifname ens1f0 master bond0
nmcli con add type ethernet con-name bond0-slave2 ifname ens1f1 master bond0
nmcli con mod bond0 ipv4.addresses 10.10.20.11/24 ipv4.method manual
nmcli con mod bond0 802-3-ethernet.mtu 9000
nmcli con up bond0
```
交换机侧需配置 LACP 与 MTU 9000。验证：`ping -M do -s 8972 10.10.20.12`。

### 3. 存储池创建步骤

#### 3.1 前置检查
```bash
eds node list                    # 确认所有节点 online
eds disk list --node node-01     # 确认 SSD/HDD 均 available
```

#### 3.2 创建存储池
1. 登录管理控制台或 CLI。
2. 执行创建命令：
```bash
eds pool create \
  --name pool-prod \
  --cache-devices /dev/nvme0n1,/dev/nvme1n1 \
  --capacity-devices /dev/sd[b-k] \
  --cache-mode writeback \
  --pg-num 128
```
3. 参数说明：
   - `--cache-mode`：`writeback` 提升写性能，`writethrough` 更安全。
   - `--pg-num`：PG 数 = (OSD 总数 × 100) / 副本数，取 2 的幂。10 OSD × 3 节点 = 30 OSD，三副本时 PG 建议 1024。
4. 验证状态：
```bash
eds pool status pool-prod
# 期望输出 HEALTH_OK，且 active+clean 占比 100%
```

### 4. 副本策略与条带配置

#### 4.1 副本策略
- **三副本**：默认策略，适用于核心业务。容忍 2 节点故障。
- **两副本**：仅用于非关键数据，容忍 1 节点故障。
- **纠删码 EC 4+2**：适用于冷数据/归档，空间利用率 66.7%，容忍 2 盘失效。

设置命令：
```bash
eds pool set pool-prod --replica 3 --min-replica 2
```
`--min-replica 2` 表示降级至 2 副本时仍可写，低于 2 则只读。

#### 4.2 条带配置
条带用于提升大文件顺序读写性能。建议：
- 条带单元 `stripe_unit`：512KB（大文件场景）或 64KB（小文件/数据库）。
- 条带宽度 `stripe_count`：4（对应 4 块 HDD 并行）。

```bash
eds pool set pool-prod --stripe-unit 524288 --stripe-count 4
```

#### 4.3 数据再平衡阈值
- 磁盘使用率 > 85% 触发再平衡。
- 单 OSD 使用率偏差 > 10% 触发回填。
- 调整命令：
```bash
eds osd set norebalance off
eds osd set backfill-full-ratio 0.85
```

#### 4.4 验证与监控
```bash
eds pool df                    # 查看容量与使用率
eds osd tree                   # 查看 OSD 分布
eds health detail              # 检查告警
```
建议接入 Prometheus，采集 `eds_pool_used_bytes`、`eds_osd_apply_latency_ms`，阈值：写延迟 > 20ms 告警。