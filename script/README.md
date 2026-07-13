# 运行脚本

## 预处理(首次)
python -c "from agent.pdf_parser import preprocess_all; preprocess_all()"
python -c "from agent.chunker import rebuild_structured_index; rebuild_structured_index()"

## A榜全量推理
python -m agent.pipeline_v43
输出: results/answer_v43.csv + results/eval_results_v42.json

## 证据导出(零token)
python _gen_evidence.py
输出: results/evidence.json
