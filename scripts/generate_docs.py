"""用 DeepSeek 生成交付场景知识库文档（扩充 sample_docs 3 篇 → 20+ 篇）。

仿 lora 项目的 gen_llm_qa_dataset.py 模式：DeepSeek 生成 + 人工抽检。
输出：data/sample_docs/ 下的 markdown 文档，覆盖桌面云/超融合/存储/安全/服务器/网络/AI 七大域。
原则：通用交付技术知识，不含厂商专有产品名与涉密信息，含具体参数/端口/命令/步骤（供检索评测）。
"""
import json
import os
import re
import time
from pathlib import Path

ENV_PATH = Path.home() / ".hermes/workspace/tasks/llm-course-interview/01-基础学习/week15graph和llm/.env"


def load_key():
    key = base = None
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY"):
                key = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("DEEPSEEK_BASE_URL"):
                base = line.split("=", 1)[1].strip().strip('"')
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not found")
    return key, base or "https://api.deepseek.com"


API_KEY, BASE_URL = load_key()

import urllib.request


def call_deepseek(prompt, max_tokens=2500, temperature=0.6, retries=3):
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  ⚠️ 第{attempt+1}次调用失败: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


# (文件名, 文档主题要求)
TOPICS = [
    ("桌面云-01-平台架构与部署.md",
     "桌面云平台整体架构与部署指南：核心组件（连接代理/虚拟桌面池/会话网关/认证服务/许可证服务）、组件间通信端口、单台/集群部署步骤、容量规划建议。写成企业交付技术文档风格，含具体端口号与步骤编号。"),
    ("桌面云-02-登录连接故障排查.md",
     "桌面云登录与连接故障排查手册：登录超时、连接断开、认证失败三类现象的排查步骤（网络延迟检查、会话网关状态、证书有效期、许可证余量），含具体排查命令与阈值建议。"),
    ("桌面云-03-性能调优指南.md",
     "桌面云性能调优指南：操作卡顿、鼠标延迟、画面冻结的常见原因（虚拟桌面配额不足/网络带宽不足/存储IO瓶颈/CPU超分比过高），给出每用户带宽建议与配额检查方法。"),
    ("桌面云-04-外设重定向配置.md",
     "桌面云外设重定向配置与排障：USB设备、打印机、读卡器无法重定向的原因（外设策略未放行/客户端版本过旧/驱动不兼容），配置步骤与验证方法。"),
    ("桌面云-05-GPU虚拟化配置.md",
     "GPU虚拟化（vGPU）配置指南：vGPU类型划分、NVIDIA授权服务器配置（默认端口7070）、推荐显卡型号（A10/A16/L20等vGPU支持卡）、存储网络建议（万兆以上）、vGPU驱动安装步骤。"),
    ("超融合-01-集群架构与部署.md",
     "超融合集群架构与部署指南：计算存储融合架构、节点角色（管理/计算/存储）、网络平面划分（业务/存储/管理）、部署前检查清单、多节点集群初始化步骤。"),
    ("超融合-02-节点故障排查.md",
     "超融合节点故障排查手册：节点离线/宕机/掉线的处理流程（先看物理机状态→网卡/固件→存储网络→集群状态），节点恢复后的数据重建与自愈检查。"),
    ("超融合-03-存储集群运维.md",
     "超融合存储集群运维指南：存储池降级、副本不足、OSD异常的处理，副本策略配置（至少2副本、重要数据3副本），容量预警阈值与扩容建议。"),
    ("超融合-04-集群扩容与迁移.md",
     "超融合集群在线扩容与虚拟机迁移指南：新增节点步骤、存储池扩容、虚拟机在线迁移（vMotion类）前提条件、迁移前后校验清单。"),
    ("存储-01-分布式存储EDS部署.md",
     "分布式存储（EDS类）部署指南：存储节点规划（SSD缓存+HDD容量）、网络要求（万兆存储网络）、存储池创建步骤、副本策略与条带配置。"),
    ("存储-02-性能优化.md",
     "分布式存储性能优化指南：SSD缓存层配置、智能分层策略、条带大小选择、小文件合并、热点数据识别，性能基线测试方法。"),
    ("存储-03-快照与备份恢复.md",
     "存储快照与备份恢复操作指南：快照创建/保留策略、备份任务配置、数据恢复演练步骤、恢复一致性检查清单。"),
    ("安全-01-防火墙策略配置与排障.md",
     "防火墙策略配置与排障手册：策略优先级规则、端口放行规划、策略不生效的排查（策略优先级错误/规则顺序冲突/会话老化），含典型业务端口清单。"),
    ("安全-02-VPN远程接入排障.md",
     "VPN与远程接入故障排查手册：拨号失败、隧道中断、IPSec断开的原因（证书失效/隧道参数不匹配/出口带宽打满），排查命令与证书续期流程。"),
    ("安全-03-等保合规建设.md",
     "企业等级保护（等保）合规建设指南：安全区域划分、访问控制策略、日志审计要求、边界防护设备部署，合规检查清单。"),
    ("服务器-01-硬件故障排查.md",
     "服务器硬件故障排查手册：电源报警、风扇异响、指示灯异常的判断（电源老化/风扇损坏/主板告警），硬件更换流程与维修窗口建议。"),
    ("服务器-02-Linux系统排障.md",
     "Linux服务器系统运维排障手册：操作系统蓝屏/引导失败/内核panic的处理（驱动冲突/磁盘损坏/内核升级失败），常见命令（dmesg、journalctl）与系统盘修复步骤。"),
    ("网络-01-数据中心网络规划.md",
     "数据中心网络规划指南：核心-汇聚-接入三层架构、VLAN划分、端口聚合、高可用（堆叠/VRRP类）设计、带宽规划与冗余策略。"),
    ("网络-02-业务割接方案.md",
     "业务割接与变更管理方案：割接前准备（回退方案/备份/窗口规划）、割接步骤、业务验证清单（连通性/性能/业务功能）、失败回退流程。"),
    ("AI-01-企业私有化大模型部署.md",
     "企业私有化大模型部署指南：模型选型（7B/14B参数量级）、推理框架（vLLM类，PagedAttention显存优化）、量化方案（INT4/INT8）、GPU资源规划、API网关与权限控制。"),
    ("AI-02-RAG知识库建设与评估.md",
     "RAG知识库建设与效果评估指南：文档清洗与分块策略（语义分块/父子块）、混合检索（向量+BM25+RRF融合）、重排（CrossEncoder）、幻觉控制（Prompt约束/引用溯源/低置信拒答）、RAGAS评估指标（Faithfulness/Hit Rate）。"),
    ("AI-03-工单智能处理Agent落地.md",
     "工单智能处理Agent落地指南：工单分类初筛→多Worker并行诊断→技能匹配分派→低置信人工复核的流程设计，评测集构建（领域知识构造可复现样例）、A/B对比验证方法。"),
]


def generate_one(filename, requirement):
    prompt = f"""你是企业云计算交付团队的资深技术文档工程师。请撰写一篇中文技术文档，主题要求如下：

{requirement}

写作要求：
1. 结构清晰：用 markdown 标题分章节（## 一级、### 二级）
2. 内容具体：包含真实可用的端口号、命令、参数阈值、步骤编号，不能空洞
3. 风格：企业交付/运维技术文档口吻，直接、务实
4. 长度：600-1200 字
5. 只输出文档正文，不要额外解释

直接输出 markdown 文档内容："""
    print(f"🔄 生成 {filename} ...")
    content = call_deepseek(prompt)
    content = content.strip()
    # 去掉可能的 ```markdown 包裹
    if content.startswith("```"):
        content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content)
    if not content.startswith("#"):
        title = filename.replace(".md", "").replace("-", " ")
        content = f"# {title}\n\n{content}"
    return content


def main():
    out_dir = Path(__file__).resolve().parents[1] / "data" / "sample_docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok, fail = 0, 0
    for filename, req in TOPICS:
        target = out_dir / filename
        if target.exists():
            print(f"⏭️  {filename} 已存在，跳过")
            ok += 1
            continue
        try:
            content = generate_one(filename, req)
            target.write_text(content, encoding="utf-8")
            print(f"✅ {filename} ({len(content)} 字)")
            ok += 1
        except Exception as e:
            print(f"❌ {filename} 失败: {e}")
            fail += 1
        time.sleep(1)
    print(f"\n完成: 成功 {ok}, 失败 {fail}, 输出目录 {out_dir}")


if __name__ == "__main__":
    main()
