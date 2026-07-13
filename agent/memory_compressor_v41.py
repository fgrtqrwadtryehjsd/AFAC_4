"""V41 创新点③: 双Agent记忆管家 — 重要性打分 + 三档分层压缩

## 用户创新点③原文
设计双Agent架构. Agent A 主干问答; Agent B(后台管家)对每条记忆打重要性分.
按遗忘曲线优先向"得分低+发生早"的开刀:
  - 最不重要的: 直接 Truncate(截断)
  - 次重要的: 转换为一句话摘要
  - 含有关键财务数据的: 保留原貌

## 赛题适配 (单题独立, 无多轮)
Agent B 用规则实现 (零 token, 非 LLM), 对文档内容三档处理:
  - TIER_KEEP (保留原貌): 题干/选项关键词+数字命中的段落 → 取 full_text ±400字完整段落
  - TIER_SUMMARY (摘要): 命中关键词但非关键数字 → 一句话摘要 (取首句)
  - TIER_TRUNCATE (截断): 无命中 → 丢弃

## 关键修正 (V40 失败教训)
V40 把关键数据压成单行碎片 → 模型看不全 → 保守少选 → 完全匹配0分.
V41 关键数据**保留完整原貌段落** (带前后文), 模型看到完整证据 → 敢选全.

## 零 token
纯规则 (字符串定位 + 窗口截取), 不调 LLM, 不调 embedding.
"""
import re
from collections import defaultdict


# 三档重要性
TIER_KEEP = "keep"        # 保留原貌: 关键数据完整段落
TIER_SUMMARY = "summary"  # 摘要: 相关但非关键
TIER_TRUNCATE = "truncate"  # 截断: 无关


class ImportanceScorer:
    """Agent B 规则版: 给文档段落打重要性分.

    评分维度 (创新点③ 重要性分数):
    - 命中选项关键词 + 关键数字: 最高分 (TIER_KEEP)
    - 命中选项关键词无数字: 中分 (TIER_SUMMARY)
    - 无命中: 0分 (TIER_TRUNCATE)
    """

    def __init__(self):
        # 财务科目关键词 (用于选项claim分解)
        # 含精确财务表标志短语 (只在财务表出现, 不在致辞命中)
        self.fin_subjects = [
            "营业收入", "营业总收入", "营收", "归母净利润",
            "归属于上市公司股东的净利润", "净利润", "扣非净利润",
            "经营活动产生的现金流量净额", "现金流量净额", "经营活动",
            "投资活动", "筹资活动",
            "研发投入占营业收入", "研发投入金额", "研发投入", "研发费用",
            "毛利率", "净利率", "资产负债率", "每股收益",
            "基本每股收益", "加权平均净资产收益率", "净资产", "总资产",
            "现金分红", "利润分配", "派发", "股份回购", "回购",
            "同比增长", "同比下降",
        ]
        self.contract_subjects = [
            "转股价格", "向下修正", "赎回", "回售", "违约",
            "资产减值", "业绩补偿", "票面利率", "信用评级",
            "主体评级", "债项评级", "发行规模", "担保", "增信",
        ]
        self.ins_subjects = [
            "身故保险金", "现金价值", "保险金额", "基本保险金额",
            "已交保费", "账户价值", "等待期", "犹豫期", "免责",
            "保单贷款", "借款", "退保", "年金", "分红", "受益人",
        ]

    def subjects_for_domain(self, domain):
        if domain == "financial_reports":
            return self.fin_subjects
        elif domain == "financial_contracts":
            return self.contract_subjects
        elif domain == "insurance":
            return self.ins_subjects
        return self.fin_subjects

    def decompose_option_claims(self, question):
        """创新点①配套: 把每个选项分解为 claim (关键词+数字).

        Returns:
            {option_key: {"keywords": [...], "numbers": [...]}}
        """
        domain = question.get("domain", "")
        subjects = self.subjects_for_domain(domain)
        options = question.get("options", {})
        claims = {}
        for ok, otext in options.items():
            kws = [s for s in subjects if s in otext]
            # 也从题干补关键词
            qtext = question.get("question", "")
            # 数字: 金额/百分比/年份
            nums = re.findall(r"\d[\d,]*(?:\.\d+)?\s*(?:%|亿|万|元|倍|天|日|月|年)?", otext + " " + qtext)
            nums = [n.strip() for n in nums if len(n.strip()) >= 2]
            claims[ok] = {"keywords": kws, "numbers": nums, "text": otext}
        return claims


class ThreeTierCompressor:
    """创新点③: 三档分层压缩器."""

    def __init__(self, scorer=None):
        self.scorer = scorer or ImportanceScorer()

    def compress(self, question, doc_index, max_chars=12000):
        """对单题所有文档做三档压缩.

        Returns:
            evidence: str, 压缩后证据 (原貌段落 + 摘要, 倒U重排由调用方做)
            stats: dict, 三档统计
        """
        doc_ids = question.get("doc_ids", [])
        domain = question.get("domain", "")
        claims = self.scorer.decompose_option_claims(question)

        # 收集所有 claim 的关键词和数字 (用于全文扫描)
        all_kws = set()
        all_nums = set()
        for c in claims.values():
            all_kws.update(c["keywords"])
            all_nums.update(c["numbers"])

        keep_segments = []   # [(score, doc_id, segment_text, claim_tag)]
        summary_segments = []  # [(score, doc_id, summary_text)]

        for did in doc_ids:
            full = doc_index.get_doc_full_text(did) or ""
            if not full:
                continue

            # 扫描全文, 找关键词命中位置
            hit_positions = []  # [(pos, kw, score)]
            for kw in all_kws:
                start = 0
                while True:
                    pos = full.find(kw, start)
                    if pos < 0:
                        break
                    hit_positions.append((pos, kw, 3))
                    start = pos + len(kw)
                    if len(hit_positions) > 200:  # 防爆
                        break

            # 关键数字命中位置 (更高分, 创新点③关键财务数据)
            for num in all_nums:
                n_core = re.sub(r"[^\d]", "", num)
                if len(n_core) < 3:
                    continue
                for m in re.finditer(re.escape(num[:max(len(num), 3)]), full):
                    hit_positions.append((m.start(), num, 5))

            # 主动定位"财务数据表"密集区 (含%或大金额的段落, 高分)
            # 这是 V40 失败的修复: 头部致辞含关键词但无数据, 财务表在后部
            for m in re.finditer(r"\d[\d,]*\.\d+\s*%|\d{3,},\d{3}", full):
                pos = m.start()
                # 检查附近是否有财务科目词 (确认是财务表不是无关数字)
                nearby = full[max(0, pos - 200):pos + 200]
                if any(kw in nearby for kw in all_kws):
                    hit_positions.append((pos, "财务数据", 6))

            if not hit_positions:
                # 无命中: 取文档头部作为摘要 (TIER_SUMMARY, 不完全丢弃)
                head = full[:1500].replace("\n", " ")
                summary_segments.append((1, did, f"[文档摘要] {head[:500]}"))
                continue

            # 按位置聚类, 合并相近命中 (限制窗口大小, 防并成超大段)
            hit_positions.sort()
            merged_windows = []  # [(start, end, max_score, kws_set)]
            i = 0
            while i < len(hit_positions):
                pos, kw, sc = hit_positions[i]
                win_start = max(0, pos - 100)
                win_end = min(len(full), pos + 350)
                win_kws = {kw}
                win_score = sc
                # 只合并 500 字内的命中 (防止不相干命中并成超大窗口)
                j = i + 1
                while j < len(hit_positions) and hit_positions[j][0] < win_end + 150:
                    p2, k2, s2 = hit_positions[j]
                    if p2 < win_end:  # 在窗口内
                        win_kws.add(k2)
                        win_score = max(win_score, s2)
                        j += 1
                    else:
                        break
                # 窗口上限 600 字 (V40 碎片太短, 但太长吃配额)
                win_end = min(win_end, win_start + 600)
                merged_windows.append((win_start, win_end, win_score, win_kws))
                i = j if j > i else i + 1

            # 选项级配额: 每个 claim 至少保留 1 个最佳窗口 (防某选项数据全丢)
            # 这是创新点①(选项claim) + ③(关键数据保留原貌) 的配合
            claim_windows = defaultdict(list)  # claim_key -> [(score, start, end, kws)]
            for win_start, win_end, sc, kws in merged_windows:
                seg_text = full[win_start:win_end]
                for ok, claim in claims.items():
                    c_kws = set(claim["keywords"])
                    c_nums = claim["numbers"]
                    # 该窗口命中了此 claim 的关键词或数字?
                    hit_kw = any(kw in seg_text for kw in c_kws)
                    hit_num = any(re.sub(r"[^\d]", "", n) and
                                  re.sub(r"[^\d]", "", n) in re.sub(r"[^\d]", "", seg_text)
                                  for n in c_nums if len(re.sub(r"[^\d]", "", n)) >= 3)
                    if hit_kw or hit_num:
                        # 加分: 同时命中关键词+数字 = 关键财务数据
                        boost = 2 if (hit_kw and hit_num) else 0
                        claim_windows[ok].append((sc + boost, win_start, win_end, kws, hit_kw and hit_num))

            # 每 claim 取 top-2 窗口 (保证每选项有数据, 又不过多)
            seen_starts = set()
            for ok, wins in claim_windows.items():
                wins.sort(key=lambda x: -x[0])
                added = 0
                for sc, ws, we, kws, is_key in wins:
                    if added >= 2:
                        break
                    # 去重: 起点相近的窗口
                    if any(abs(ws - s) < 200 for s in seen_starts):
                        continue
                    seen_starts.add(ws)
                    seg = full[ws:we].replace("\n", " ")
                    kw_tag = ",".join(sorted(kws)[:3])
                    keep_segments.append((sc, did, seg, kw_tag))
                    added += 1

        # 按分数排序, 取 top (TIER_KEEP)
        keep_segments.sort(key=lambda x: -x[0])

        # 组装: TIER_KEEP 原貌段落 (倒U重排在 context_surgeon 做)
        # 这里先按 doc 分组, 标注清晰
        parts = []
        total = 0
        per_doc_keep = defaultdict(int)
        seen_text = set()  # 去重 (不同窗口可能重叠)

        for sc, did, seg, kw_tag in keep_segments:
            if total >= max_chars:
                break
            # 去重: 段落前 80 字唯一
            key = seg[:80]
            if key in seen_text:
                continue
            seen_text.add(key)
            if total + len(seg) > max_chars:
                seg = seg[: max_chars - total]
            parts.append({
                "tier": TIER_KEEP,
                "doc_id": did,
                "score": sc,
                "text": seg,
                "tag": kw_tag,
            })
            per_doc_keep[did] += 1
            total += len(seg)

        # 补摘要 (TIER_SUMMARY) 填充剩余空间
        for sc, did, sm in summary_segments:
            if total >= max_chars:
                break
            if total + len(sm) > max_chars:
                sm = sm[: max_chars - total]
            parts.append({
                "tier": TIER_SUMMARY,
                "doc_id": did,
                "score": sc,
                "text": sm,
                "tag": "summary",
            })
            total += len(sm)

        evidence_parts = parts  # 保留结构给倒U重排
        return evidence_parts, {
            "n_keep": sum(1 for p in parts if p["tier"] == TIER_KEEP),
            "n_summary": sum(1 for p in parts if p["tier"] == TIER_SUMMARY),
            "per_doc_keep": dict(per_doc_keep),
            "total_chars": total,
            "claims": claims,
        }
