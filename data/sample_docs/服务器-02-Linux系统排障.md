## Linux服务器系统运维排障手册：操作系统蓝屏/引导失败/内核panic的处理

### 一、故障分类与快速定位

Linux 无“蓝屏”概念，对应表现为 **kernel panic**、**systemd 启动失败**、**GRUB 引导中断**。按现象分三类：

| 现象 | 典型原因 | 优先排查 |
|------|----------|----------|
| 启动卡在 GRUB / `grub rescue>` | 引导分区损坏、磁盘掉线 | `lsblk`、`fdisk -l` |
| 启动中 kernel panic | 驱动冲突、根分区挂载失败 | `dmesg`、`journalctl -k -b -1` |
| 升级内核后无法启动 | initramfs 缺失、驱动未重建 | GRUB 旧内核回退 |

### 二、核心诊断命令

#### 2.1 dmesg（内核环形缓冲区）
```bash
dmesg -T --level=err,warn          # 带时间戳，仅错误/警告
dmesg | grep -iE "panic|oops|call trace|ata[0-9]+.*error"
dmesg -T | tail -n 200             # 最近200行
```
阈值参考：`ata` 报错超过 5 次/分钟 → 磁盘或线缆故障；`nvme` 出现 `I/O timeout` → 盘体或背板问题。

#### 2.2 journalctl（systemd 日志）
```bash
journalctl -k -b -1                # 上一次启动的内核日志
journalctl -p err -b               # 本次启动所有错误
journalctl --since "2025-01-01 08:00" --until "2025-01-01 09:00"
journalctl -u systemd-modules-load.service   # 驱动加载服务
```
关键字段：`Failed to start`、`Dependency failed`、`Kernel command line`。

#### 2.3 端口与磁盘快速检查
```bash
ss -tlnp | grep -E ':(22|80|443|3306|6379)'   # 确认关键服务端口
smartctl -a /dev/sda | grep -E "Reallocated|Pending|Uncorrectable"
# 阈值：Reallocated_Sector_Ct > 10 或 Pending > 0 → 立即换盘
```

### 三、系统盘修复步骤（引导失败场景）

#### 步骤 1：挂载救援模式
通过 IPMI/IDRAC 挂载 ISO，或使用 U 盘启动，选择 **Rescue a broken system**。

#### 步骤 2：识别根分区
```bash
lsblk -f
# 输出示例：sda1 ext4 /boot, sda2 LVM, sda3 xfs /
vgchange -ay            # 激活 LVM
```

#### 步骤 3：检查并修复文件系统
```bash
fsck -y /dev/sda1       # /boot 分区
fsck -y /dev/mapper/vg-root
# 注意：XFS 用 xfs_repair -L /dev/sda3（-L 强制清日志，慎用）
```

#### 步骤 4：重建 GRUB 与 initramfs
```bash
mount /dev/mapper/vg-root /mnt
mount /dev/sda1 /mnt/boot
for i in dev proc sys; do mount --bind /$i /mnt/$i; done
chroot /mnt
grub2-install /dev/sda
grub2-mkconfig -o /boot/grub2/grub.cfg
dracut -f /boot/initramfs-$(uname -r).img $(uname -r)   # RHEL/CentOS
update-initramfs -u -k all                               # Debian/Ubuntu
exit; reboot
```

### 四、内核升级失败回退

1. 重启进入 GRUB，选择 **Advanced options → 旧内核**。
2. 确认旧内核可用后，锁定版本：
```bash
yum versionlock kernel-4.18.0-477.el8    # RHEL
apt-mark hold linux-image-5.15.0-91-generic  # Ubuntu
```
3. 清理异常新内核：
```bash
rpm -qa | grep kernel-5.15.0-100
rpm -e kernel-5.15.0-100 --nodeps
```
4. 若 GRUB 默认项错误，修改 `/etc/default/grub` 中 `GRUB_DEFAULT=saved`，执行 `grub2-mkconfig`。

### 五、驱动冲突处理

现象：加载某模块后 panic，日志含 `BUG: unable to handle kernel NULL pointer`。

```bash
lsmod | grep <模块名>
modprobe -r <模块名>              # 卸载
echo "blacklist <模块名>" >> /etc/modprobe.d/blacklist.conf
dracut -f                          # 重建 initramfs 使黑名单生效
```
若冲突发生在升级后，对比 `journalctl -k -b -1 | grep <模块>` 与当前版本，回退驱动包：
```bash
dnf downgrade kmod-<驱动>
```

### 六、交付验收检查项

- `dmesg -T --level=err` 无新增 I/O 错误
- `journalctl -p err -b` 无 failed unit
- `smartctl -H /dev/sda` 返回 PASSED
- `grub2-editenv list` 默认内核与预期一致
- 重启 3 次均正常进入多用户模式（`systemctl get-default` = multi-user.target）