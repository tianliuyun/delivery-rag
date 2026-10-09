## GPU虚拟化（vGPU）配置指南

本文面向企业云计算交付与运维人员，说明 NVIDIA vGPU 的类型划分、授权服务器部署、推荐硬件选型、存储网络要求及驱动安装流程。适用于 VMware vSphere、KVM/QEMU、Citrix Hypervisor 等主流虚拟化平台。

### 一、vGPU 类型划分

NVIDIA vGPU 按使用模式分为三类，交付前需与业务确认选型：

- **vGPU（时间片虚拟化）**：单块物理 GPU 切分为多个 vGPU 实例，适合 VDI、办公、轻量图形场景。实例规格如 `1B`、`2B`、`4B`、`8B`（B 系列面向虚拟桌面）。
- **vGPU（MIG 模式）**：基于 A100/A30 等 Ampere 及以上架构，通过 MIG 将 GPU 切分为独立实例，隔离性更强，适合 AI 推理多租户。
- **GPU 直通（Passthrough）**：整卡分配给单台虚拟机，无虚拟化开销，适合训练、高性能计算。

选型建议：VDI 用 B 系列；AI 推理用 A 系列 vGPU 或 MIG；训练用直通。

### 二、NVIDIA 授权服务器（vGPU License Server）配置

vGPU 必须联网或指向内网授权服务器，否则实例会在 20 分钟后降频或停止渲染。

1. 下载 NVIDIA License Server 安装包（如 `NVIDIA-ls-linux-2023.11.0`）。
2. 安装依赖并执行安装：
   ```bash
   sudo ./setup.bin -i console
   ```
3. 默认服务端口为 **7070**，Web 管理端口为 **8080**。确认防火墙放行：
   ```bash
   sudo firewall-cmd --permanent --add-port=7070/tcp
   sudo firewall-cmd --permanent --add-port=8080/tcp
   sudo firewall-cmd --reload
   ```
4. 浏览器访问 `https://<license-server-ip>:8080`，登录后上传从 NVIDIA 获取的 `.bin` 授权文件。
5. 在虚拟机内 vGPU 驱动配置中指定授权服务器地址，例如：
   ```bash
   echo "LicenseServer=7070@10.0.0.100" >> /etc/nvidia/gridd.conf
   ```
6. 验证授权状态：
   ```bash
   nvidia-smi -q | grep -i license
   ```
   输出应显示 `License Status: Licensed`。

### 三、推荐显卡型号

以下型号官方支持 vGPU，交付中常见：

- **NVIDIA A10**：24GB 显存，适合 VDI、云游戏、AI 推理，单卡可切分多实例。
- **NVIDIA A16**：4×16GB，专为虚拟桌面设计，密度高，适合大规模 VDI。
- **NVIDIA L20**：48GB 显存，Ada 架构，适合生成式 AI 推理与图形工作站。
- **NVIDIA A40**：48GB，适合专业图形与中等训练。
- **NVIDIA H20**：面向大模型推理，合规场景可选。

注意：消费级 GeForce 卡不支持 vGPU 授权，不可用于生产虚拟化。

### 四、存储与网络建议

- **网络**：vGPU 节点与授权服务器、存储之间建议 **万兆（10GbE）以上**，AI 推理场景推荐 25GbE 或 100GbE。启用 SR-IOV 或 RDMA 可降低延迟。
- **存储**：虚拟机镜像与 vGPU 缓存盘建议使用全闪存存储，IOPS 不低于 50000。AI 场景建议 NVMe over Fabric。
- **授权服务器**：需与 vGPU 节点网络互通，延迟低于 10ms，建议部署双实例做 HA。

### 五、vGPU 驱动安装步骤

以 Linux 虚拟机（Ubuntu 22.04）为例：

1. 在虚拟化平台为虚拟机添加 vGPU 设备，选择对应 profile（如 `A10-4Q`）。
2. 启动虚拟机，确认 PCI 设备可见：
   ```bash
   lspci | grep -i nvidia
   ```
3. 安装依赖：
   ```bash
   sudo apt update && sudo apt install -y build-essential dkms
   ```
4. 下载对应版本的 vGPU Guest Driver（如 `NVIDIA-Linux-x86_64-535.161.05-vgpu-kvm.run`）。
5. 禁用 Nouveau：
   ```bash
   echo -e "blacklist nouveau\noptions nouveau modeset=0" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
   sudo update-initramfs -u
   ```
6. 安装驱动：
   ```bash
   sudo sh NVIDIA-Linux-x86_64-535.161.05-vgpu-kvm.run --dkms -s
   ```
7. 配置授权服务器并重启服务：
   ```bash
   sudo systemctl restart nvidia-vgpud
   sudo systemctl restart nvidia-vgpu-mgr
   ```
8. 验证：
   ```bash
   nvidia-smi
   ```
   应显示 vGPU 型号、显存及授权状态。

Windows 虚拟机步骤类似，使用对应 `.exe` 驱动，安装后在 NVIDIA 控制面板中填入授权服务器 `7070@<ip>`。

### 六、交付检查清单

- 授权服务器 7070 端口可达，授权文件已加载
- vGPU profile 与业务负载匹配
- 网络 ≥ 万兆，存储 IOPS 达标
- 驱动版本与虚拟化平台、vGPU 管理器版本一致
- `nvidia-smi` 显示 Licensed，无降频告警