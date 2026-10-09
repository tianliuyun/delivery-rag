## VPN与远程接入故障排查手册

### 1. 适用范围
本手册适用于企业分支/移动办公场景下的 IPSec VPN、SSL VPN 与远程拨号接入故障定位，覆盖拨号失败、隧道中断、IPSec 断开三类高频问题。

### 2. 常见故障原因与判定

#### 2.1 证书失效
- 现象：IKE 第一阶段协商失败，日志出现 `certificate expired`、`no suitable certificate`。
- 判定：检查本地证书有效期，确认 CA 链完整。
- 命令：
```bash
openssl x509 -in vpn.crt -noout -dates
openssl verify -CAfile ca.crt vpn.crt
```
- 阈值：证书剩余有效期小于 30 天应触发续期工单。

#### 2.2 隧道参数不匹配
- 现象：`NO_PROPOSAL_CHOSEN`、`INVALID_ID_INFORMATION`、`phase2 negotiation failed`。
- 常见不匹配项：
  - IKE 版本：IKEv1 vs IKEv2
  - 加密算法：AES-128 vs AES-256
  - 哈希：SHA1 vs SHA256
  - DH 组：Group2 vs Group5/14
  - 预共享密钥或对端 ID
- 命令：
```bash
ipsec statusall
ipsec whack --status
journalctl -u strongswan -n 200
```
- 核查：两端 `proposal`、`lifetime`（默认 28800s）、`DPD` 必须一致。

#### 2.3 出口带宽打满
- 现象：隧道频繁重连，丢包率高，`DPD timeout`。
- 判定：出口链路利用率持续 > 85%，或丢包 > 3%。
- 命令：
```bash
sar -n DEV 1 5
iftop -i eth0
tc -s qdisc show dev eth0
```
- 处置：限速非关键业务，QoS 保障 UDP 500/4500 与 ESP 流量。

### 3. 排查步骤

1. 确认物理链路与路由可达：
```bash
ping -c 4 <VPN_PEER_IP>
traceroute <VPN_PEER_IP>
```
2. 检查端口放通：UDP 500、UDP 4500、ESP（协议号 50）、SSL VPN TCP 443。
3. 查看 IKE/IPSec 状态：
```bash
ipsec status
ipsec statusall
```
4. 抓包定位：
```bash
tcpdump -i eth0 -n -vv 'udp port 500 or udp port 4500'
tcpdump -i eth0 -n -vv 'proto 50'
```
5. 核对证书与参数，必要时重启服务：
```bash
systemctl restart strongswan
systemctl restart ipsec
```

### 4. 证书续期流程

1. 生成 CSR：
```bash
openssl req -new -newkey rsa:2048 -nodes -keyout vpn.key -out vpn.csr
```
2. 提交 CA 签发，获取新证书 `vpn.crt` 与 CA 链。
3. 备份旧证书：
```bash
cp /etc/ipsec.d/certs/vpn.crt /etc/ipsec.d/certs/vpn.crt.bak
```
4. 替换证书并校验：
```bash
openssl verify -CAfile ca.crt vpn.crt
```
5. 重载服务：
```bash
systemctl reload strongswan
```
6. 验证隧道：
```bash
ipsec statusall | grep ESTABLISHED
```
7. 观察 10 分钟，确认无 DPD 超时与重协商失败。

### 5. 预防建议
- 证书到期前 30 天自动告警。
- 两端参数纳入配置基线，变更需双人复核。
- 出口带宽设置 80% 告警阈值，关键 VPN 流量标记 DSCP EF。