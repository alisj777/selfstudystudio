# selfstudystudio：雅思 AI 自学系统

雅思自学 AI tutor：知识图谱 + 三路 RAG（语料 / 评分 / 话题）+ 多 Agent 测评链。
设计文档：https://claude.ai/artifact/XP7BJaHq2FuYpyEH9LQppf

当前进度：**M1 数据与知识图谱**（Agent 与评分模型在后续里程碑）。

## 快速开始

```bash
uv sync                      # 核心依赖
uv sync --extra anthropic    # 使用 Claude；或 --extra openai（OpenAI / Ollama / vLLM）
uv sync --extra embeddings   # 可选：bge-m3 等语义嵌入
cp .env.example .env

uv run ielts-tutor build                                   # 加载图谱、校验、建三路索引 -> var/
uv run ielts-tutor search "traffic in big cities" -k 5     # 三路融合检索
uv run ielts-tutor search "over-use of linkers" --source rubric --skill writing
uv run ielts-tutor tag-corpus                              # 用 LLM 给未标注的题目打标签，输出待审核队列
uv run ielts-tutor import-scored essays.csv --text-col Essay --overall-col Overall --name kaggle
uv run pytest
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `data/ielts/kg/*.yaml` | 种子本体：技能、评分维度与 band 描述、题型、写作结构、话题、语法点、错误类型、听力主题 |
| `data/ielts/corpus/*.jsonl` | 语料 RAG 数据（`CorpusChunk`）；示例为自编练习题 |
| `data/ielts/scoring/*.jsonl` | 评分 RAG 的已评分样本（`ScoredSample`），用 `import-scored` 生成 |
| `src/ielts_tutor/kg/` | 图谱 schema（边的类型约束）、YAML 加载与校验、NetworkX 存储、导出 Cypher（Neo4j）、LLM 打标签 |
| `src/ielts_tutor/rag/` | 嵌入、BM25 + 向量混合索引、三路检索器、RRF 融合、建索引 |
| `src/ielts_tutor/llm/` | 可切换 LLM：Anthropic SDK / OpenAI 兼容接口 / 测试用 Fake |

## 注意

- `01_skills_criteria.yaml` 中的 band 描述是**改写摘要**（`source: paraphrase`），目前只有 Band 5–8；用于评分前请对照官方公开评分描述核实并补全。
- 默认的 `hashing` 嵌入只做字面匹配，用于测试和离线演示；实际效果请切换到 `sentence-transformers`（如 `BAAI/bge-m3`）后重新 `build`。
- 真题语料版权归各方所有，只作内部检索，不要提交到仓库。
