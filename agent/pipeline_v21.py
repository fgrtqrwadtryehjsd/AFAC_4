"""V21 流水线 — V20 + fc 锚词扩展"""
import os, json
from collections import Counter
from agent.config import QUESTIONS_DIR, RESULTS_DIR, TOKEN_BUDGET
from agent.qwen_client import QwenClient
from agent.indexer import DocumentIndex
from agent.reasoner_v21 import ReasoningAgentV21
from agent.postprocessor import generate_answer_csv_token_stats


def load_questions(split: str = "A") -> list:
    questions = []
    qa_dir = os.path.join(QUESTIONS_DIR, f"group_{split.lower()}")
    if not os.path.exists(qa_dir): return questions
    for fn in sorted(os.listdir(qa_dir)):
        if fn.endswith('.json'):
            with open(os.path.join(qa_dir, fn), 'r', encoding='utf-8') as f:
                questions.extend(json.load(f))
    return questions


def run_a_board():
    print('=' * 60)
    print('AFAC2026 V21 — V20 + fc 锚词扩展')
    print(f'qwen-plus | budget {TOKEN_BUDGET:,}')
    print('=' * 60)

    questions = load_questions('A')
    print(f'加载 {len(questions)} 题')

    di = DocumentIndex(); di.load()
    qwen = QwenClient()
    agent = ReasoningAgentV21(qwen, di, None, token_budget=TOKEN_BUDGET)

    print('\n开始推理...\n')
    results = []
    for i, q in enumerate(questions):
        stats = qwen.get_token_stats()
        if stats['total_tokens'] > TOKEN_BUDGET * 0.95:
            print(f'⚠️ token 上限')
            for rq in questions[i:]:
                results.append({'qid': rq['qid'], 'answer': ''})
            break
        print(f'[{i+1}/{len(questions)}]', end='')
        r = agent.answer_question(q)
        results.append({'qid': q['qid'], 'answer': r['answer']})
        print(f' {q["qid"]} ({q.get("domain","")[:4]}/{q.get("answer_format","")}) → {r["answer"]} [证据{r["evidence_chars"]//1000}K/{r["total_doc_chars"]//1000}K]')

    print('\n' + '=' * 60)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    stats = qwen.get_token_stats()
    total = stats['total_tokens']
    ts = max(0, min(1, (TOKEN_BUDGET - total) / TOKEN_BUDGET))

    out = generate_answer_csv_token_stats(
        results, stats['prompt_tokens'], stats['completion_tokens'], total)

    import shutil
    v21_csv = os.path.join(RESULTS_DIR, 'answer_v21.csv')
    shutil.copy(out, v21_csv)
    print(f'  备份 V21 → {v21_csv}')

    agent.save_cot_trails()

    ans = [r['answer'] for r in results if r['answer']]
    dist = Counter(ans).most_common(15)
    single = [a for a in ans if len(a)==1]
    sd = {c: single.count(c) for c in 'ABCD'}

    print(f'\n📊 V21 摘要:')
    print(f'  有效: {len([r for r in results if r["answer"]])}/{len(questions)}')
    print(f'  Token: {total:,}')
    print(f'  TokenScore: {ts:.4f}')
    print(f'  调用: {stats["call_count"]}')
    print(f'  分布: {dist}')
    print(f'  单选: A={sd.get("A",0)} B={sd.get("B",0)} C={sd.get("C",0)} D={sd.get("D",0)}')
    print(f'\n  📊 V20 参照: TokenScore=0.2285 | Score=43.81')


if __name__ == '__main__':
    run_a_board()
