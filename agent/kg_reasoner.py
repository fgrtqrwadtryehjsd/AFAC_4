"""V42: 知识图谱压缩记忆 + 多轮图算法推理

用户方向: "把知识以图的形式压缩检索, 多轮用图算法回复".
图 = 压缩记忆 (节点+边 ≪ 60K 原文); 多轮图遍历 = 逐选项断言沿边验证.

## 节点
- claim 节点: 每选项的原子断言 (关键词+数字), 复用 ImportanceScorer.decompose_option_claims
- evidence 节点: 题干关键词全局定位段 + 每选项专属 grep 段 (修注意力瓶颈, res_a_020 已验证)
- fact 节点: 从 evidence 抽 数字/年份 (fin/contract 域可补 DocumentCard metrics)

## 边 (规则判定, 非 LLM, 零 token)
- claim —hard— evidence: 选项数字逐字命中 或 实体关键词逐字命中
- claim —soft— evidence: 同义命中 (synonym_expander: 贷款↔借款, 不得↔不在) 但原词不在
- 矛盾判定留 Round 3 Qwen (需语义, 规则只做粗筛)

## 多轮 (Phase B 实现)
- Round1 定位子图 (0 token): claim 种子 → evidence/fact 邻居
- Round2 规则分类 (0 token): hard+无矛盾→SELECTED; 矛盾→REJECTED; 仅soft/无→UNCERTAIN
- Round3 Qwen 裁决: 只对 UNCERTAIN, 喂结构化 per-claim 边 (非自由文本, 稳定性关键)
- Round4 完整性补全 (multi): UNCERTAIN 定向 re-grep + 再裁决, 修漏选/多选

## 合规
纯规则构图零 token, 非"模型"; 最终答案来自 Qwen 当题推理; 同义/矛盾边由规则定, 非预计算 LLM 判断.
"""
import re
from agent.memory_compressor_v41 import ImportanceScorer
from agent.reasoner_multiround import extract_keywords, locate_sections
from agent.synonym_expander import SYNONYM_MAP
from agent.reasoner_v20 import _take_head


def _digits(s):
    """提取字符串中的数字串 (用于数字逐字比对)"""
    return re.sub(r"[^\d]", "", s)


def _num_unit_match(num_str, text):
    """数字+单位一起匹配, 防止 coincidental 假匹配 (如 5-10倍 假匹配 5-10万元).

    旧法 _digits 后比子串: '5-10倍'→'510' 假匹配 '5-10万元'→'510'.
    新法要求 数字核心+单位 一起出现: '5-10倍' 需原文 '5-10...倍', '5-10万元' 不含'倍' → 不匹配.
    """
    n = (num_str or "").strip()
    m = re.match(r"([\d\.,\-~]+)\s*(.*)", n)
    if not m:
        return False
    core, unit = m.group(1), m.group(2).strip()
    if not core:
        return False
    core_re = re.escape(core)
    if unit:
        unit_re = re.escape(unit[:3])
        pat = core_re + r"\s{0,2}" + unit_re
    else:
        # 无单位: 数字后须非数字 (防 123 假匹配 1234)
        pat = core_re + r"(?=\D|$)"
    return bool(re.search(pat, text))


# research 域 subject 词 (ImportanceScorer.subjects_for_domain 默认回退 fin_subjects, 对研报覆盖弱)
_RESEARCH_SUBJECTS = [
    "市场规模", "市场容量", "市场份额", "市占率", "市场占有率",
    "复合增长率", "CAGR", "年复合增长率", "同比增长", "增速",
    "渗透率", "毛利率", "净利率", "营收", "营业收入", "净利润",
    "保费", "保费收入", "保费贡献率", "赔付", "赔付率",
    "龙头", "头部企业", "竞争格局", "集中度",
    "预期", "预计", "预测", "展望",
]

# 边关键词过滤: 起始为虚词/助词的片段丢弃 (extract_keywords 的 3-8 字碎片噪声)
_KW_BAD_START = set("了的为和中与或及是对从向将被把给到上下内外前后这那以及等")


def _clean_gen_kws(kws):
    """过滤 extract_keywords 的噪声碎片: 去虚词起首/过短/纯数字"""
    out = []
    seen = set()
    for k in kws:
        if not k or len(k) < 3:
            continue
        if k[0] in _KW_BAD_START:
            continue
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


class GraphBuilder:
    """零 token 规则构图: claim/evidence/fact 节点 + hard/soft 边"""

    # 综合金融术语表 (跨域并集): 选项可能提他域概念 (如fc题提"资产负债率"属fin域词)
    # 用于 subject grep, 避免域专属词表漏匹配
    _ALL_SUBJECTS = None

    def __init__(self, doc_index, doc_cards=None):
        self.di = doc_index
        self.cards = doc_cards or {}
        self.scorer = ImportanceScorer()
        if GraphBuilder._ALL_SUBJECTS is None:
            sc = self.scorer
            terms = set(sc.fin_subjects) | set(sc.contract_subjects) | set(sc.ins_subjects)
            terms |= set(_RESEARCH_SUBJECTS)
            GraphBuilder._ALL_SUBJECTS = sorted(terms, key=len, reverse=True)  # 长词优先

    def _option_subjects(self, opt_text):
        """从选项文本抽综合术语 (跨域), 长词优先防短词吃长词"""
        return [s for s in self._ALL_SUBJECTS if s in opt_text]

    def build(self, question, max_evidence_per_doc=8000):
        qid = question["qid"]
        domain = question.get("domain", "")
        doc_ids = question.get("doc_ids", [])
        options = question.get("options", {})
        q_text = question.get("question", "")

        # --- claim 节点 ---
        # 边关键词 (clean): 域 subject 词; 检索关键词 (broad): 含 gen_kws 用于 grep 召回
        raw_claims = self.scorer.decompose_option_claims(question)
        is_research = (domain == "research")
        claims = {}
        for ok, c in raw_claims.items():
            subject_kws = list(c["keywords"])
            if is_research:
                # research 域补 subject 词 (默认回退 fin_subjects 覆盖弱)
                subject_kws = [s for s in _RESEARCH_SUBJECTS if s in c["text"]] + subject_kws
                subject_kws = list(dict.fromkeys(subject_kws))
            gen_kws = _clean_gen_kws(extract_keywords(c["text"]))  # broad, 供检索
            # 边关键词 = clean subject + 清洗后的 gen (去重保序)
            edge_kws = list(dict.fromkeys(subject_kws + gen_kws))
            # 数字: 仅选项文本, 去重 (decompose_option_claims 混入题干数字, 会污染边)
            nums_raw = re.findall(r"\d[\d,]*(?:\.\d+)?\s*(?:%|亿|万|元|倍|股|年|个工作日|个月|日)?", c["text"])
            nums = list(dict.fromkeys(n.strip() for n in nums_raw if len(_digits(n)) >= 2))
            claims[ok] = {
                "text": c["text"],
                "keywords": edge_kws,         # 边判定用 (clean)
                "subject_keywords": subject_kws,  # Round3 优先展示
                "gen_keywords": gen_kws,      # 检索用 (broad)
                "numbers": nums,              # 选项专属数字 (clean)
            }

        # --- evidence 节点 ---
        evidence_nodes = []
        eid = 0
        for did in doc_ids:
            text = self.di.get_doc_full_text(str(did)) or ""
            if not text:
                continue
            # (a) 题干+全选项关键词全局定位段
            all_kw = extract_keywords(q_text + " " + " ".join(str(v) for v in options.values()))
            expanded_kw = set(all_kw)
            for kw in all_kw:
                if kw in SYNONYM_MAP:
                    expanded_kw.update(SYNONYM_MAP[kw])
            kw_seg = locate_sections(text, list(expanded_kw), max_evidence_per_doc // 2)
            if kw_seg:
                evidence_nodes.append({"id": f"e{eid}", "doc_id": str(did), "pos": -1,
                                       "text": kw_seg, "tag": "query_kw", "opt": None})
                eid += 1
            # (b) 每选项专属 grep (修注意力瓶颈: 每选项关键段紧贴选项)
            for ok in sorted(options):
                opt_text = options[ok]
                opt_kws = extract_keywords(opt_text)
                opt_exp = set(opt_kws)
                for kw in opt_kws:
                    if kw in SYNONYM_MAP:
                        opt_exp.update(SYNONYM_MAP[kw])
                num_kws = [k for k in opt_exp if re.search(r"\d", k)]
                other_kws = [k for k in opt_exp if not re.search(r"\d", k)]
                search_kws = num_kws[:5] + other_kws[:5]
                opt_seg = self._option_evidence(text, search_kws, max_chars=2500)
                if opt_seg:
                    evidence_nodes.append({"id": f"e{eid}", "doc_id": str(did), "pos": -1,
                                           "text": opt_seg, "tag": f"opt{ok}", "opt": ok})
                    eid += 1

        # --- fact 节点 (从 evidence 抽数字/年份) ---
        fact_nodes = []
        fid = 0
        seen_facts = set()
        for ev in evidence_nodes:
            t = ev["text"]
            for m in re.finditer(r"\d[\d,]*(?:\.\d+)?\s*(?:%|亿|万|元|倍|股|年|个工作日|个月|日)?", t):
                val = m.group().strip()
                dval = _digits(val)
                if len(dval) >= 2 and (dval, ev["doc_id"]) not in seen_facts:
                    seen_facts.add((dval, ev["doc_id"]))
                    fact_nodes.append({"id": f"f{fid}", "type": "number", "value": val,
                                       "digits": dval, "doc_id": ev["doc_id"], "ev_id": ev["id"]})
                    fid += 1

        # --- 边 (规则判定) ---
        edges = []
        ev_text_digits = {ev["id"]: _digits(ev["text"]) for ev in evidence_nodes}
        for ok, claim in claims.items():
            c_kws = claim["keywords"]
            c_nums = [n for n in claim["numbers"] if len(_digits(n)) >= 2]
            for ev in evidence_nodes:
                evtext = ev["text"]
                # hard: 选项数字+单位一起命中 (防 5-10倍 假匹配 5-10万元)
                hard_nums = [n for n in c_nums if _num_unit_match(n, evtext)]
                # hard: 选项实体关键词逐字命中
                kw_hits = [kw for kw in c_kws if kw and kw in evtext]
                # soft: 同义命中 (原词不在, 同义词在)
                soft_hits = []
                for kw in c_kws:
                    if kw in SYNONYM_MAP and kw not in evtext:
                        for syn in SYNONYM_MAP[kw]:
                            if syn != kw and syn in evtext:
                                soft_hits.append((kw, syn))
                                break
                etype = None
                if hard_nums or kw_hits:
                    etype = "hard"
                elif soft_hits:
                    etype = "soft"
                if etype:
                    edges.append({
                        "src": f"claim{ok}", "dst": ev["id"], "type": etype,
                        "hard_nums": hard_nums, "kw_hits": kw_hits,
                        "soft": soft_hits, "doc": ev["doc_id"], "opt_tag": ev["tag"],
                    })

        return {
            "qid": qid, "domain": domain,
            "claims": claims,
            "evidence": evidence_nodes,
            "facts": fact_nodes,
            "edges": edges,
            "evidence_chars": sum(len(e["text"]) for e in evidence_nodes),
        }

    def _option_evidence(self, text, search_kws, max_chars=2500):
        """每选项专属 grep: 优先取多关键词共现的高密度段 (避开头部单关键词噪声)

        旧法逐关键词取首2命中 → "2024年"等通用词首命中在doc头部, 淹没6.97%等稀有特定数.
        新法: 聚类合并800字内命中, 按 distinct关键词数 降序取段 → 6.97%段(4词共现)优先于头部(1词).
        """
        if not search_kws:
            return ""
        pos_kws = []  # [(pos, kw)]
        for kw in search_kws:
            if not kw:
                continue
            for m in re.finditer(re.escape(kw), text):
                pos_kws.append((m.start(), kw))
                if len(pos_kws) > 400:  # 防爆
                    break
        if not pos_kws:
            return ""
        pos_kws.sort()
        # 聚类: 合并 800 字内命中
        clusters = []  # [(start, end, distinct_kws, n_hits)]
        i = 0
        while i < len(pos_kws):
            start = max(0, pos_kws[i][0] - 300)
            end = min(len(text), pos_kws[i][0] + 700)
            kws_in = {pos_kws[i][1]}
            n = 1
            j = i + 1
            while j < len(pos_kws) and pos_kws[j][0] - pos_kws[j - 1][0] < 800:
                end = min(len(text), pos_kws[j][0] + 700)
                kws_in.add(pos_kws[j][1])
                n += 1
                j += 1
            clusters.append((start, end, len(kws_in), n))
            i = j if j > i else i + 1
        # 按 (distinct_kws desc, n_hits desc) 取段
        clusters.sort(key=lambda c: (-c[2], -c[3]))
        parts = []
        used = []
        for start, end, dk, n in clusters:
            if sum(len(p) for p in parts) >= max_chars:
                break
            # 去重: 与已取段重叠 >60% 跳过
            overlap = False
            for s, e, _, _ in used:
                if not (end <= s or start >= e):
                    ov = min(end, e) - max(start, s)
                    if ov > 0.6 * min(end - start, e - s):
                        overlap = True
                        break
            if overlap:
                continue
            parts.append(text[start:end])
            used.append((start, end, dk, n))
        result = "\n---\n".join(parts)[:max_chars]
        return result if result else ""


# ============ Phase B: GraphTraverser 多轮图推理 ============

# V42 multi 基线证据上限 (降token): V35 是 60K (fc/fin) / 45K每doc (res/ins 无上限).
# V42 统一 cap 到 30K 总量, 靠 add-only 审计补回漏选. 审计只能帮不能害(vs 30K基线).
V42_MULTI_BASELINE = 30000


def build_evidence_v42(doc_index, domain, doc_ids, max_evidence=V42_MULTI_BASELINE):
    """V42 multi 证据: 全域统一 cap (不再给 res/ins 45K/doc 特权), 降 token.

    复用 V35 域提取器 (head+锚词), 仅缩小 per_doc. 审计 Round3 补回漏选.
    """
    from agent.reasoner_v35 import V35_DOMAIN_EXTRACTORS, extract_evidence_regulatory
    extractor = V35_DOMAIN_EXTRACTORS.get(domain, extract_evidence_regulatory)
    n_docs = max(1, len(doc_ids))
    per_doc = max_evidence // n_docs
    evidence = ""
    for did in doc_ids:
        t = doc_index.get_doc_full_text(str(did)) or ""
        if not t:
            continue
        seg = extractor(t, max_chars=per_doc)
        evidence += f"\n=== 文档 {did} ===\n{seg}\n"
    return evidence

# Round3 逐选项结构化裁决 prompt (演化自 PROMPT_MULTI_GRAPH, 输入是结构化 per-claim 边)
_ROUND3_MULTI = """你的任务: 依据每选项的专属证据与图边, 判断该选项是否正确.

## 锚定状态
{state}

## 选项与专属证据 (每选项证据已紧贴选项, 避免跨选项错配)
{option_blocks}

## 判断流程 (证据-断言关联图)
对每选项:
1. 拆原子断言: [核心]=主体/关键数值/行为主干; [边缘]=修饰用词
2. 证据类型: 硬=选项数值/主体在专属证据逐字出现; 软=语义等价(同义/上下位/默认指代); 无=专属证据无对应
3. 硬矛盾(否决): 数值不符/主体错位/方向相反/年份张冠李戴/单位错位(每股vs每10股)/计算值与选项冲突
4. 软差异(容忍, 不否决): 同义用词(借款=保单贷款)/等价表述(不得=不在)/默认指代(现金分红=年度现金分红)
5. 判定: 核心断言有证据(硬或软)且无硬矛盾 -> 选; 核心断言无证据或有硬矛盾 -> 不选
6. 漏选与过选都0分, 只在有证据支持时选; 但有硬/软证据支持时必须选, 不可因"用词不完全一致"漏选

## 输出格式 (必须对A/B/C/D全部分析, 不许早停)
选项A: 断言[核心] | 证据类型(硬/软/无) | 硬矛盾(无/有) | 判定(选/不选)
选项B: ...
选项C: ...
选项D: ...
最终答案: <按字母序拼接所有选的选项>
"""

# Round4 完整性补全 (multi 漏选安全网, 证据门控: 硬须单位匹配, 软须明确同义)
_ROUND4_MULTI = """初判答案: {draft}
已选: {selected}
未选: {unselected}

## 任务: 对每个未选选项, 仅基于其专属证据与图边, 判断是否应补充 (修漏选)
{option_blocks}

## 判断规则 (收紧: 防过选)
应补充 ONLY IF 满足下列之一:
- 硬证据: 选项核心数值+单位在专属证据逐字出现 (如选项"6.97%"需原文"6.97%", 非"6.97万"; "5-10倍"需原文"5-10倍", 非"5-10万元")
- 软证据: 明确语义等价 — 同一概念的不同说法 (如"借款"="保单贷款", "不得"="不在"), 须是确凿同义, 非仅关键词相近

不补充 IF:
- 仅关键词巧合命中但语义不符 (如"5-10倍"命中"5-10万元"车价, 单位不同)
- 主体错位: 选项说A公司/A产品, 证据是B公司/B产品
- 年份/单位/数值不符 (年份张冠李戴, 每股vs每10股, 数值微差)
- 拿不准时, 不补充 (宁可不补, 不可过选)

## 输出格式
未选选项X: 专属证据中<原文引用或"无"> | 判定: 补充/不补充 | 理由
...
最终答案: <按字母序拼接(已选+应补充)>
"""


class GraphTraverser:
    """多轮图算法推理: Round2规则分类 → Round3 Qwen裁决 → Round4完整性补全"""

    def __init__(self, qwen, doc_index):
        self.qwen = qwen
        self.di = doc_index
        self.builder = GraphBuilder(doc_index)
        try:
            from agent.anchor_state_v41 import AnchorState
            self.anchor = AnchorState()
        except Exception:
            self.anchor = None

    # --- Round 2: 规则预分类 (0 token) ---
    def classify(self, graph):
        status = {}
        for ok in graph["claims"]:
            edges = [e for e in graph["edges"] if e["src"] == f"claim{ok}"]
            hard = [e for e in edges if e["type"] == "hard"]
            soft = [e for e in edges if e["type"] == "soft"]
            if hard:
                status[ok] = "HAS_HARD"
            elif soft:
                status[ok] = "SOFT_ONLY"
            else:
                status[ok] = "NO_EVIDENCE"
        return status

    # --- 渲染: 每选项专属证据 + 图边标签 (hit-centered snippets, 非blob head) ---
    def _option_block(self, graph, ok, ev_chars=1500, max_snippets=4):
        claim = graph["claims"].get(ok, {})
        edges = [e for e in graph["edges"] if e["src"] == f"claim{ok}"]
        hard = [e for e in edges if e["type"] == "hard"]
        soft = [e for e in edges if e["type"] == "soft"]

        # 收集 hit-centered snippets: 优先特定数(含%/亿/万/倍/长数字), 后通用词/年份
        def _spec_score(h):
            d = _digits(h)
            if re.search(r"%|亿|万|倍|元", h):
                return 3  # 最特定
            if len(d) >= 5:
                return 2
            return 1 if d else 0

        candidates = []  # [(spec_score, ev_text, hit, doc, tag)]
        for e in hard + soft:
            ev = next((x for x in graph["evidence"] if x["id"] == e["dst"]), None)
            if not ev:
                continue
            text = ev["text"]
            hits = list(e.get("hard_nums", [])) + list(e.get("kw_hits", [])) \
                + [s[1] for s in e.get("soft", [])]
            for h in hits:
                if not h:
                    continue
                pos = text.find(h)
                if pos < 0:
                    continue
                candidates.append((_spec_score(h), text, h, e["doc"], e["opt_tag"], pos))
        # 按特定度降序, 去重相近位置
        candidates.sort(key=lambda c: -c[0])
        snippets = []
        used_pos = []  # (text_id, pos)
        for sc, text, h, doc, tag, pos in candidates:
            if len(snippets) >= max_snippets:
                break
            if any(text is t and abs(pos - p) < 300 for t, p in used_pos):
                continue
            used_pos.append((text, pos))
            start = max(0, pos - 200)
            end = min(len(text), pos + 450)
            snippets.append(f"[{doc}@{tag}] 命中\"{h}\": ...{text[start:end].strip()}...")
        ev_text = "\n".join(snippets)[:ev_chars] if snippets else "(无专属证据)"

        # 图边标签 (供模型快速定位关键数/词)
        hard_nums = sorted({n for e in hard for n in e.get("hard_nums", [])})
        hard_kws = sorted({k for e in hard for k in e.get("kw_hits", [])})
        soft_pairs = sorted({f"{a}={b}" for e in soft for a, b in e.get("soft", [])})
        docs = sorted({e["doc"] for e in edges})
        label_parts = []
        if hard_nums:
            label_parts.append(f"硬数字{hard_nums[:4]}")
        if hard_kws:
            label_parts.append(f"硬词{hard_kws[:4]}")
        if soft_pairs:
            label_parts.append(f"软等价{soft_pairs[:3]}")
        label = "; ".join(label_parts) if label_parts else "无命中"
        return f"选项{ok}: {claim.get('text','')}\n  图边@{','.join(docs) if docs else '-'}: {label}\n  专属证据:\n{ev_text}"

    # --- Round 3: Qwen 逐选项裁决 ---
    def adjudicate(self, question, graph, system):
        from agent.postprocessor import extract_answer_from_response
        state_text = ""
        if self.anchor:
            try:
                _, state_text = self.anchor.build(question, self.di)
            except Exception:
                state_text = ""
        options = question.get("options", {})
        blocks = "\n\n".join(self._option_block(graph, ok) for ok in sorted(options))
        prompt = _ROUND3_MULTI.format(state=state_text or "(无)", option_blocks=blocks)
        try:
            r = self.qwen.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=4096, timeout=180)
            raw = r["content"]
        except Exception as e:
            raw = f"ERR:{e}"
        ans = extract_answer_from_response(raw, "multi")
        return ans, raw, len(prompt)

    # --- Round 4: 完整性补全 (multi 漏选安全网) ---
    def completeness(self, question, graph, draft, system, unselected_only_chars=1800):
        from agent.postprocessor import extract_answer_from_response
        options = question.get("options", {})
        draft_set = set(draft) if draft else set()
        unselected = sorted(set(options.keys()) - draft_set)
        if not unselected:
            return draft, "", 0
        blocks = "\n\n".join(
            self._option_block(graph, ok, ev_chars=unselected_only_chars)
            for ok in unselected)
        prompt = _ROUND4_MULTI.format(
            draft=draft, selected=sorted(draft_set), unselected=unselected,
            option_blocks=blocks)
        try:
            r = self.qwen.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=2048, timeout=120)
            raw = r["content"]
        except Exception as e:
            raw = f"ERR:{e}"
        ans = extract_answer_from_response(raw, "multi")
        return ans, raw, len(prompt)


class ReasoningAgentV42:
    """V42: multi 走 KG-MultiRound; tf/mcq 暂留 V35 (单变量, 铁律)"""

    def __init__(self, qwen, doc_index, vector_indexer=None,
                 token_budget=5_000_000, model="qwen3.6-plus"):
        # 复用 V35 (tf/mcq 路径 + _post_process + cot_trails)
        from agent.reasoner_v35 import ReasoningAgentV35
        self.v35 = ReasoningAgentV35(qwen, doc_index, vector_indexer, token_budget, model)
        self.qwen = qwen
        self.di = doc_index
        self.traverser = GraphTraverser(qwen, doc_index)
        self.cot_trails = []

    def answer_question(self, question):
        af = question.get("answer_format", "mcq")
        if af == "multi":
            return self._answer_multi_v42(question)
        return self.v35.answer_question(question)

    def _v42_post_multi(self, ans):
        """V42 multi 后处理: 允许 ABCD 4 字母 (不截断到3).

        V20 _post_process 截到3字母是 V5 时代防 ABCD 泛滥的守门, 但会砍掉真值 ABCD 的题
        (ins_a_017/019 审计正确补D→ABCD 被截回 ABC). V42 审计是证据门控的, 过选风险可控,
        故不截断, 仅过滤非法字符.
        """
        if not ans:
            return ""
        chars = sorted(set(c for c in ans.upper() if c in "ABCD"))
        return "".join(chars) if chars else ""

    def _answer_multi_v42(self, question):
        from agent.reasoner_v20 import DOMAIN_SYSTEM, PROMPT_MULTI
        from agent.postprocessor import extract_answer_from_response
        from agent.reasoner_v35 import build_evidence_v35
        qid = question["qid"]
        domain = question.get("domain", "")
        system = DOMAIN_SYSTEM.get(domain, "")
        q_text = question["question"]
        options = question.get("options", {})
        doc_ids = question.get("doc_ids", [])

        # Round 1: 构图 (0 token) — 供 Round3 审计用
        graph = self.traverser.builder.build(question)
        status = self.traverser.classify(graph)  # 诊断用

        # Round 2: V35 基线 — 广域 60K 证据 + V20 prompt (保 V35 正确率, 防回归)
        # PhaseB v6 教训: 30K 压缩致正确率崩 (12/25), 审计窄grep无法补回广域证据丢失.
        # 压缩与正确率 1:1 交换, 净分持平. 故保 60K 基线.
        evidence = build_evidence_v35(self.di, domain, doc_ids, self.v35.MULTI_MAX_EVIDENCE)
        prompt = PROMPT_MULTI.format(
            evidence=evidence, question=q_text,
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
        base_ev = len(evidence)

        # Round 3: 图审计 add-only (修漏选, 仅增不删, 防回归)
        # 对 draft 未选选项, 用图专属证据 + 证据门控宽松判定是否补充.
        final = draft
        raw4 = ""
        p4 = 0
        if draft:
            ans4, raw4, p4 = self.traverser.completeness(question, graph, draft, system)
            ans4 = self._v42_post_multi(ans4)
            # 仅当 Round3 是 draft 的超集 (只增不删) 且更长才采纳, 否则保留 draft
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
            "strategy": "v42_v35plus_audit",
        })
        return {
            "qid": qid, "answer": final,
            "evidence_chars": base_ev,
            "total_doc_chars": total_doc_chars,
        }

    def save_cot_trails(self, path=None):
        import os, json
        from agent.config import RESULTS_DIR
        path = path or os.path.join(RESULTS_DIR, "eval_results_v42.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cot_trails, f, ensure_ascii=False, indent=2)
        print(f"  V42 COT trails -> {path}")

