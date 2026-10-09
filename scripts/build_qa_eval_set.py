"""从交付知识库文档构建 QA 评测集（DeepSeek 生成 + golden_context 命中判据）。

方法（仿 lora 的 gen_llm_qa_dataset.py）：
1. 每篇文档整篇作为上下文，一次调用 DeepSeek 生成 3-5 条 QA 对
   （比按章节块调用快 5 倍：25 篇 ≈ 25 次调用）
2. 每条 QA 带 golden_context（文档原文子串，用于检索命中判定）
3. 输出 data/qa_eval_set.jsonl：question / answer / source / golden_context
"""
import json
import re
import sys
import time
import urllib.request
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


def call_deepseek(prompt, max_tokens=2500, temperature=0.4, retries=3):
    payload = json.dumps({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=240) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  ⚠️ 第{attempt+1}次调用失败: {e}")
            if attempt == retries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def gen_qa_for_doc(filename: str, doc_text: str, n_per_doc: int = 4) -> list:
    prompt = f"""你是企业知识库 QA 构建专家。下面是一篇企业交付技术文档，请基于文档内容生成 {n_per_doc} 条高质量问答对。

要求：
1. 问题必须能从文档中找到明确答案（不能问文档之外的内容）
2. 问题要有检索区分度：包含具体参数、端口、命令、阈值、步骤或产品术语（模拟工程师真实提问）
3. 答案必须引用文档中的具体信息，一到两句话
4. 每个 golden_context 必须是文档中的原文连续子串（用于检索命中判定），直接从文档原文复制，不能改写
5. 问题之间要覆盖文档不同章节，不要集中在同一段

输出 JSON 对象，格式：
{{"qa_pairs": [{{"question": "...", "answer": "...", "golden_context": "..."}}]}}

文档文件名：{filename}
文档内容：
{doc_text[:6000]}
"""
    raw = call_deepseek(prompt)
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError(f"无法解析 JSON: {raw[:300]}")
    obj = json.loads(m.group(0))
    items = obj.get("qa_pairs", [])
    # 校验 golden_context 在原文中
    norm_doc = re.sub(r"\s+", "", doc_text)
    valid = []
    for it in items:
        it["source"] = filename
        gc = it.get("golden_context", "")
        if gc and re.sub(r"\s+", "", gc) in norm_doc:
            valid.append(it)
        else:
            print(f"  ⚠️ golden_context 不在原文: {gc[:40]}")
    return valid


def main():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    docs_dir = data_dir / "sample_docs"
    out_path = data_dir / "qa_eval_set.jsonl"

    docs = sorted(docs_dir.glob("*.md"))
    print(f"📄 共 {len(docs)} 篇文档")

    all_items = []
    seen = set()
    for i, doc in enumerate(docs):
        text = doc.read_text(encoding="utf-8")
        try:
            items = gen_qa_for_doc(doc.name, text)
            for it in items:
                q = it["question"].strip()
                if q in seen:
                    continue
                seen.add(q)
                all_items.append(it)
            print(f"✅ [{i+1}/{len(docs)}] {doc.name}: {len(items)} 条")
        except Exception as e:
            print(f"❌ [{i+1}/{len(docs)}] {doc.name} 失败: {e}")
        time.sleep(0.5)

    with open(out_path, "w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\n✅ 评测集生成完成: {len(all_items)} 条 -> {out_path}")
    from collections import Counter
    cnt = Counter(it["source"] for it in all_items)
    for src, n in cnt.most_common():
        print(f"  {src}: {n}")


if __name__ == "__main__":
    main()
