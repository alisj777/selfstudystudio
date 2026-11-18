# selfstudystudio：雅思 AI 自学系统

雅思自学 AI tutor：知识图谱 + 三路 RAG（语料 / 评分 / 话题）+ 多 Agent 测评链。
设计文档：https://claude.ai/artifact/XP7BJaHq2FuYpyEH9LQppf

当前进度：**M1 数据与知识图谱**（Agent 与评分模型在后续里程碑）。

## 快速开始

```bash
uv sync                      # 核心依赖
uv sync --extra anthropic    # 使用 Claude；或 --extra openai（OpenAI / Ollama / vLLM）
uv sync --extra embeddings   # 语义嵌入（默认 BAAI/bge-small-en-v1.5）
cp .env.example .env         # 填入 API key；.env 中的值优先于 shell 里导出的同名变量

uv run ielts-tutor build                                   # 加载图谱、校验、建三路索引 -> var/
uv run ielts-tutor search "traffic in big cities" -k 5     # 三路融合检索
uv run ielts-tutor search "over-use of linkers" --source rubric --skill writing
uv run ielts-tutor eval --show-misses                       # 检索评测：对比各种检索方案
uv run ielts-tutor tag-corpus                              # 用 LLM 给未标注的题目打标签，输出待审核队列
uv run ielts-tutor import-scored essays.csv --text-col Essay --overall-col Overall --name kaggle
uv run pytest
```

## 目录

| 路径 | 内容 |
| --- | --- |
| `data/ielts/kg/*.yaml` | 种子本体：技能、评分维度与 band 描述、题型、写作结构、话题、语法点、错误类型、听力主题 |
| `data/ielts/corpus/*.jsonl` | 语料 RAG 数据（`CorpusChunk`）；示例为自编练习题 |
| `data/ielts/glossary.yaml` | 雅思术语别名表（含中文），用于扩展关键词检索的查询 |
| `data/ielts/eval/retrieval_queries.yaml` | 检索评测集（40 条，含 3 条中文） |
| `data/ielts/scoring/*.jsonl` | 评分 RAG 的已评分样本（`ScoredSample`），用 `import-scored` 生成 |
| `src/ielts_tutor/kg/` | 图谱 schema（边的类型约束）、YAML 加载与校验、NetworkX 存储、导出 Cypher（Neo4j）、LLM 打标签 |
| `src/ielts_tutor/rag/` | 嵌入、BM25 + 向量混合索引、三路检索器、RRF 融合、建索引 |
| `src/ielts_tutor/llm/` | 可切换 LLM：Anthropic SDK / OpenAI 兼容接口 / 测试用 Fake |

## 检索方案与评测

三路检索都用"关键词（BM25，词干化）+ 语义向量"混合，两边结果用 RRF 融合。别名表只扩展关键词那一侧的查询，语义向量那一侧用原始查询。`uv run ielts-tutor eval` 的结果（40 条查询，2026-11-18）：

| 方案 | hit@1 | hit@3 | MRR |
| --- | --- | --- | --- |
| 字面哈希 | 0.57 | 0.68 | 0.67 |
| 字面哈希 + 别名表 | 0.68 | 0.85 | 0.78 |
| bge-small-en | 0.65 | 0.88 | 0.77 |
| **bge-small-en + 别名表（默认）** | **0.80** | **1.00** | **0.89** |

评测集很小，而且查询和别名表出自同一作者，数字会偏乐观；扩充评测集时，最好收集真实学生的提问。

## 注意

- `01_skills_criteria.yaml` 中的 band 描述是**改写摘要**（`source: paraphrase`），目前只有 Band 5–8；用于评分前请对照官方公开评分描述核实并补全。
- 换嵌入模型后必须重新 `build`（索引会记录建索引时用的模型，不一致会报错）。bge-small-en 只懂英文，中文提问目前靠别名表；需要更好的中文支持可换 `BAAI/bge-m3`。
- macOS 13 上 torch 最高只能装 2.11（2.12 起只提供 macOS 14+ 的安装包），`pyproject.toml` 已做限制。
- 真题语料版权归各方所有，只作内部检索，不要提交到仓库。
