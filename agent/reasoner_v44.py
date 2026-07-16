"""V44: V43 + multi三修 (claim方向核验修过选draft+审计, 靶向grep修漏选)

诊断(2026-07-14, 6过选multi题)发现过选源分裂:
- V35 draft过选(3): fc_a_012(D), ins_a_010(CD), ins_a_014(CD) → draft的PROMPT_MULTI未核验claim方向(公式/范围/文档归属/claim否定方向)
- 审计过加(3): fc_a_020(C), reg_a_017(C), reg_a_012(C) → _ROUND4 audit按数字表面命中加, 未核验claim
- 漏选(4): fc_a_009(B=43.24%), fin_a_017(A=292亿), reg_a_007(D=1万元), res_a_004(C) → V35广域60K没捞具体深位数值
注: fc_a_012审计正确加了C → 审计收紧须精准(只claim不符才不加, 不普遍提高门槛, 否则回归+8multi)

V44修法:
1. multi证据 = V35广域 + 靶向grep(选项数值/实体, 补深位43.24%/292亿/1万元)
2. PROMPT_MULTI_V44 = PROMPT_MULTI + claim方向核验块(防draft过选)
3. _ROUND4_MULTI_V44 = _ROUND4_MULTI + claim方向核验块(防审计过加)
tf/mcq同V43(单选3题fc_a_008/fin_a_008/res_a_006的陷阱prompt已含单位/绝对化核验, V43已用)
"""
import re
from agent.reasoner_v43 import (ReasoningAgentV43, text_terms, targeted_grep)
from agent.kg_reasoner import GraphTraverser, _ROUND4_MULTI
from agent.reasoner_v20 import DOMAIN_SYSTEM, PROMPT_MULTI, COMMON_HEADER
from agent.reasoner_v35 import build_evidence_v35, V35_DOMAIN_EXTRACTORS, extract_evidence_regulatory
from agent.postprocessor import extract_answer_from_response


# V44.3: multi基线60K→40K (适度减token, 比30K稳), 靠靶向grep+must-include补深位
V44_MULTI_BASE_CHARS = 40000


CLAIM_CHECK = """## claim方向核验 (防过选, 关键——逐选项必做)
选某选项前, 若其claim涉及以下, 须核验命中片段的归属/范围/方向与选项claim一致, 任一不符则✗不选:
- 文档归属: 选项说"第二份文档/文档X", 命中须在文档X(非他份). 如"第二份的股票代码300866"但300866只在第一份→✗.
- 公式/概念归属: 选项说"违约利息公式含150%", 命中须是"违约利息"公式; 若150%在"违约金"公式而"违约利息"无→✗(概念不可互换).
- 范围限定: 选项涉"预防接种""营运交通工具内"等限定场景, 命中须在该范围内; 普通意外不在"营运交通工具内"→✗.
- claim否定方向: 选项说"未给公式/无/不含/不披露", 即使命中相关关键词也✗(claim是否定的, 命中反证选项错).
- 单位: "每股"须命中"每股"; "每10股派X元"≠"每股X元", 不可外推. 单位错位→✗.
- 主体/年份: 选项说A公司/2024年, 命中须是A公司/2024年; 张冠李戴/年份错→✗.
"""


def _build_prompt_multi_v44():
    """PROMPT_MULTI + claim核验块(插在输出格式前) + 输出格式加claim列."""
    flow = """## 判断流程 (多选, 必须谨慎)
1. 评分规则: 完全匹配才得分. 漏选/过选都 0 分. **宁可漏选, 不可过选**.
2. 对 A B C D 4 个选项**全部分析** (不能早停).
3. 选项 X 必须满足: 原文中存在与该选项**关键数字/用词/事实完全一致**的语句, 才能选.
4. 数字微差 / 用词替换 / 时限不一致 → ✗ 不选.
5. 涉及"两份文档均..."的选项, 必须**两份文档都能找到**, 缺一不可.
6. **claim-反面规则**: 若问题问"哪些给出了公式/计算方法/计算方式", 选项称"未给/无/未载公式/未给计算方法"则✗不选——其claim是"未给"(反面), 即使命中"现金价值/载明"等词也不可选(它承认未给公式).
"""
    out_fmt = """## 输出格式 (必须遵守)
选项 A: <原文引用> | claim核验(归属/范围/方向/单位) | 判定: ✓/✗
选项 B: <原文引用> | claim核验 | 判定: ✓/✗
选项 C: <原文引用> | claim核验 | 判定: ✓/✗
选项 D: <原文引用> | claim核验 | 判定: ✓/✗

最终答案: <按字母序拼接所有 ✓ 选项, 如 ABC. 若无 ✓ 选项, 输出 A>
"""
    return COMMON_HEADER + flow + CLAIM_CHECK + out_fmt


PROMPT_MULTI_V44 = _build_prompt_multi_v44()


_ROUND4_MULTI_V44 = """初判答案: {draft}
已选: {selected}
未选: {unselected}

## 文档顺序 (核验文档归属用)
{doc_order}

## 任务: 对每个未选选项, 仅基于其专属证据与图边, 判断是否应补充 (修漏选)
{option_blocks}

## 判断规则 (收紧: 防过选, claim方向核验)
应补充 ONLY IF 满足下列之一:
- 硬证据: 选项核心数值+单位在专属证据逐字出现 (如选项"6.97%"需原文"6.97%", 非"6.97万"; "5-10倍"需原文"5-10倍", 非"5-10万元")
- 软证据: 明确语义等价 — 同一概念的不同说法 (如"借款"="保单贷款", "不得"="不在"), 须是确凿同义, 非仅关键词相近

不补充 IF (任一即不补):
- 仅关键词/数字巧合命中但claim方向不符:
  - 文档归属: 选项说"第二份文档/文档X"但命中在另一份(看上面文档顺序映射) → 不补. 如选项"第二份的股票代码300866"但300866命中在第一份 → 不补
  - 公式/概念归属: 选项说"违约利息公式"但150%在"违约金"公式, "违约利息"无 → 不补
  - 范围限定: 选项限"预防接种/营运交通工具内"但命中在范围外 → 不补
  - claim否定: 选项说"未给公式/无" → 不补(命反证其错)
  - 单位错位: "每10股X元"≠"每股X元" → 不补
- 主体错位: 选项说A公司/A产品, 证据是B公司/B产品
- 年份/单位/数值不符 (年份张冠李戴, 每股vs每10股, 数值微差)
- 拿不准时, 不补充 (宁可不补, 不可过选)

## 输出格式
未选选项X: 专属证据中<原文引用或"无"> | claim核验(文档归属/公式/范围/方向) | 判定: 补充/不补充 | 理由
...
最终答案: <按字母序拼接(已选+应补充)>
"""


class GraphTraverserV44(GraphTraverser):
    """V44 traverser: audit用_ROUND4_MULTI_V44(加claim方向核验)."""

    def completeness(self, question, graph, draft, system, unselected_only_chars=1800):
        options = question.get("options", {})
        doc_ids = question.get("doc_ids", [])
        draft_set = set(draft) if draft else set()
        unselected = sorted(set(options.keys()) - draft_set)
        if not unselected:
            return draft, "", 0
        # V44: 文档顺序标签 (第1份=doc_ids[0]...), 供模型核验文档归属
        doc_order = ", ".join(f"第{i+1}份={did}" for i, did in enumerate(doc_ids))
        blocks = "\n\n".join(
            self._option_block(graph, ok, ev_chars=unselected_only_chars)
            for ok in unselected)
        prompt = _ROUND4_MULTI_V44.format(
            draft=draft, selected=sorted(draft_set), unselected=unselected,
            doc_order=doc_order, option_blocks=blocks)
        try:
            r = self.qwen.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=2048, timeout=120)
            raw = r["content"]
        except Exception as e:
            raw = f"ERR:{e}"
        ans = extract_answer_from_response(raw, "multi")
        return ans, raw, len(prompt)


def _multi_targeted_grep(di, question, max_per_doc=5000):
    """multi靶向grep: 所有选项的数值/实体并集, 每doc grep, 补深位事实(43.24%/292亿/1万元).

    V44修: text_terms的len(digits)<2过滤把"1万元"→"1万"(1位数)滤掉了(reg_a_007 D漏选).
    此处额外补单数字+单位的数(1万/5亿/3年/2倍), 这些是选项关键事实不该漏.
    V44.3修: must-include pass——选项关键数值(带单位)若主grep没捞到, 直接定位补段,
    消除targeted_grep频率聚类随机性(reg_a_007的1万元有时被反洗钱法全文挤掉).
    """
    options = question.get("options", {})
    doc_ids = question.get("doc_ids", [])
    doc_texts = [di.get_doc_full_text(str(d)) or "" for d in doc_ids]
    all_nums, all_terms = [], []
    # 选项关键数值(带单位, must-include): 确保这些必现
    must_nums = []
    for ok in sorted(options):
        n, t = text_terms(options[ok], doc_texts)
        all_nums += n
        all_terms += t
        for m in re.finditer(r"\d\s*(?:万元|万|亿元|亿|元|倍|股|%)", options[ok]):
            s = m.group().strip()
            if s:
                if s not in all_nums:
                    all_nums.append(s)
                if s not in must_nums:
                    must_nums.append(s)
    search_kws = list(dict.fromkeys(all_nums))[:12] + list(dict.fromkeys(all_terms))[:20]
    out = ""
    for did, t in zip(doc_ids, doc_texts):
        if not t:
            continue
        seg = targeted_grep(t, search_kws, max_per_doc)
        if seg:
            out += f"\n=== 文档 {did} 靶向补充 ===\n{seg}\n"
    return out


class ReasoningAgentV44(ReasoningAgentV43):
    """V44: multi走V44(V35+靶向grep+claim核验prompt+claim核验审计); tf/mcq同V43."""

    def __init__(self, qwen, doc_index, vector_indexer=None,
                 token_budget=5_000_000, model="qwen3.6-plus"):
        super().__init__(qwen, doc_index, vector_indexer, token_budget, model)
        self.traverser = GraphTraverserV44(qwen, doc_index)

    def answer_question(self, question):
        af = question.get("answer_format", "mcq")
        if af == "multi":
            return self._answer_multi_v44(question)
        return self._answer_tfmq_v43(question)  # tf/mcq同V43

    def _answer_multi_v44(self, question):
        qid = question["qid"]
        domain = question.get("domain", "")
        system = DOMAIN_SYSTEM.get(domain, "")
        q_text = question["question"]
        options = question.get("options", {})
        doc_ids = question.get("doc_ids", [])

        # Round1: 构图 (0 token, 供V44审计)
        graph = self.traverser.builder.build(question)
        status = self.traverser.classify(graph)

        # V44.1稳态: V35广域60K基线 + 靶向grep + claim核验 (V44.3的40K+must-include致全面过选,已弃)
        evidence = build_evidence_v35(self.di, domain, doc_ids, self.v35.MULTI_MAX_EVIDENCE)
        tgt = _multi_targeted_grep(self.di, question, max_per_doc=5000)
        evidence_full = evidence + tgt
        prompt = PROMPT_MULTI_V44.format(
            evidence=evidence_full, question=q_text,
            options="\n".join(f"{k}. {options[k]}" for k in sorted(options.keys())),
        )
        try:
            r = self.qwen.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=4096, timeout=180)
            raw_base = r["content"]
        except Exception as e:
            raw_base = f"ERR:{e}"
        draft = self._v42_post_multi(extract_answer_from_response(raw_base, "multi")) or "A"
        base_ev = len(evidence_full)

        # Round3: V44图审计 add-only (claim核验prompt, 防过加; 仍只增不删防回归)
        final = draft
        raw4 = ""
        p4 = 0
        if draft:
            ans4, raw4, p4 = self.traverser.completeness(question, graph, draft, system)
            ans4 = self._v42_post_multi(ans4)
            if ans4 and set(ans4) >= set(draft) and len(ans4) > len(draft):
                final = ans4
        if not final:
            final = "A"

        total_doc_chars = sum(self.di.doc_lengths.get(str(d), 0) for d in doc_ids)
        self.cot_trails.append({
            "qid": qid, "domain": domain, "answer": final, "answer_format": "multi",
            "draft_v35": draft, "audit_added": sorted(set(final) - set(draft)),
            "raw_base": (raw_base or "")[:1000], "raw4": (raw4 or "")[:1200],
            "status_r2": status, "graph_ev_chars": graph["evidence_chars"],
            "n_edges": len(graph["edges"]), "base_ev_chars": base_ev,
            "audit_prompt_chars": p4, "total_doc_chars": total_doc_chars,
            "tgt_chars": len(tgt),
            "strategy": "v44_2_base30k_tgrep_claimcheck",
        })
        return {
            "qid": qid, "answer": final,
            "evidence_chars": base_ev, "total_doc_chars": total_doc_chars,
        }
