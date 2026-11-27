# selfstudystudio：雅思 AI 自学系统

> **🚧 开发中。** 当前代码都在 [`m1-kg-rag`](https://github.com/alisj777/selfstudystudio/tree/m1-kg-rag) 分支上，`main` 只有初始提交，暂未合并。
> 状态：听、说、读、写四个模块的 MVP 已跑通（2026-11-23），正在进入精调阶段；评分准确度还不能替代真人考官。

面向没有老师批改的雅思自学考生：在一个地方完成四科练习，客观题自动判分并给中文解析，写作和口语给出分项评分和**带依据**的改进建议，并根据每次练习的结果安排下一题。

- 产品设计文档：[docs/PRODUCT.md](docs/PRODUCT.md)（用户、场景、功能范围、产品决策、指标、路线图）
- 技术设计文档：https://claude.ai/artifact/XP7BJaHq2FuYpyEH9LQppf

## 目录

- [功能与进度](#功能与进度)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [准备数据](#准备数据)
- [使用](#使用)
- [代码结构](#代码结构)
- [开发](#开发)
- [已知局限与下一步](#已知局限与下一步)
- [技术细节](#技术细节)

## 功能与进度

| 模块 | 能做什么 | 状态 |
| --- | --- | --- |
| 写作 | 按薄弱项选真题或生成新题 → 提交作文 → 四项分（模型 + LLM 考官各 50%）、错误标注、3–6 条引用证据的建议 | ✅ MVP；评分精度待提升 |
| 听力 | 剑4–13、17 共 44 套：书页原图 + 分段音频 → 40 题答题卡 → 分数段、逐题对错、错题中文解析 | ✅ MVP |
| 阅读 | 同上，44 套学术类 | ✅ MVP；GT 阅读仅命令行 |
| 口语 | Part 1–3 真题 → 浏览器录音（Part 2 有 1 分钟准备）→ 本机转写 → 四项分 + 建议 | ✅ MVP；分数仅供参考，一律待复核 |
| 学习者档案 | 四科成绩、评分维度/题型/话题/语法点掌握度，驱动下一次出题 | ✅ 滑动平均；BKT 待做 |
| 托福 | 数据层按考试划分，接口已预留 | ⏸ 未开始 |

关键数字（详见[技术细节](#技术细节)）：

- 知识图谱 233 个节点、339 条关系
- 检索 hit@1 0.82 / MRR 0.88（40 条评测查询）
- 答案提取覆盖率约 97.5%（3,840 题）
- 中文解析 558 段，错题能找到解析的比例：听力约 59%、学术类阅读约 68%
- 写作：LLM 考官在 12 篇真考官评分作文中有 9 篇相差在 0.5 分以内；模型在留出集上 QWK 0.43（受训练标签质量限制）
- 自动化测试 66 项全部通过

## 系统架构

```
            ┌──────────────────── 网页（FastAPI + 单页）/ 命令行（Typer） ────────────────────┐
            │                                                                              │
   写作/口语 ▼                                                                    听力/阅读 ▼
  LangGraph 多 Agent 链                                                     PracticeService
  规划 → 出题 ──(学生作答)──► 作答分析 → 评分 ─(分歧>1档,一次)─► 作答分析        判分 + 分数段 + 中文解析
                                         └──► 反馈 → 学生建模
        │            │             │          │
        ▼            ▼             ▼          ▼
   知识图谱      三路 RAG：语料 / 评分 / 话题      评分模型（特征+向量+GBR）   LLM（GPT / Claude，可切换）
  (NetworkX)    BM25 + 向量混合，RRF 融合          本机 Whisper 转写 + 流利度特征
```

- **知识图谱**：题型、话题、语法点、错误类型、评分维度与 band 描述。出题选型、错误归类（LLM 报出的错误类型必须是图谱节点）、薄弱语法点推荐、掌握度追踪都基于它。
- **三路 RAG**：语料（真题，约 1,300 片段）、评分（官方描述、考官评语、已评分作文）、话题，各自检索后融合。
- **防幻觉**：错误片段必须在原文中出现；反馈必须引用本次检索到的证据 id，否则丢弃；模型与 LLM 分歧过大时自动复查，仍不一致就标记人工复核。
- **隐私与版权**：录音只在本机转写；剑桥真题、音频、提取出的答案和解析只放在本地 `private/` 目录，不进 git。

## 快速开始

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/alisj777/selfstudystudio.git
cd selfstudystudio
git checkout m1-kg-rag                 # 开发分支

uv sync --extra openai --extra embeddings --extra agents --extra web --extra scoring --extra speech
cp .env.example .env                   # 填入 API key（.env 已被 git 忽略）

uv run ielts-tutor build               # 加载图谱、校验、建三路索引 -> var/
uv run ielts-tutor serve               # http://127.0.0.1:8000
```

可选依赖：

| extra | 用途 |
| --- | --- |
| `anthropic` / `openai` | LLM 提供方（OpenAI 兼容接口也可接 Ollama / vLLM） |
| `embeddings` | 语义向量（默认 `BAAI/bge-small-en-v1.5`） |
| `ocr` | PDF 渲染与 OCR（另需 `brew install tesseract tesseract-lang`） |
| `agents` / `web` / `scoring` | LangGraph、FastAPI、scikit-learn |
| `speech` | 本机语音识别 faster-whisper（模型首次使用时自动下载） |

没有 API key 也能跑：`serve --no-llm` / `write submit --no-llm` 只用本地模型和规则，反馈会简单很多。

## 准备数据

仓库里只有种子图谱、自编示例题和评测集。真题需要你自己的剑桥 PDF 和音频（放在 `data/ielts/` 下，不会被提交）：

```bash
uv run ielts-tutor import-ielts-org                                      # ielts.org 免费官方样题与考官评分作文
uv run ielts-tutor import-cambridge "data/ielts/剑4-20真题+解析+音频"      # 题目、原文、音频、答案、页码清单
uv run ielts-tutor import-explanations "data/ielts/剑4-20真题+解析+音频"   # OCR 中文解析（约 1 小时，可中断续跑）
uv run ielts-tutor import-btnotpt                                        # 写作评分训练数据
uv run ielts-tutor train-scorer                                          # -> var/models/writing_scorer.pkl
uv run ielts-tutor build                                                 # 导入后重建索引
```

扫描版（剑14–20 的部分 PDF）暂不支持，会被跳过并提示。

## 使用

网页（`serve`）有写作、听力、阅读、口语四个标签页，右上角显示各科最近成绩。命令行：

```bash
# 写作
uv run ielts-tutor write start --topic "online education"   # --source generate 让 LLM 出新题
uv run ielts-tutor write submit <session_id> essay.txt
uv run ielts-tutor write profile

# 口语
uv run ielts-tutor speak start --part 2
uv run ielts-tutor speak submit <session_id> answer.m4a

# 听力 / 阅读判分
uv run ielts-tutor check 17 1 listening my_answers.txt      # 书号 套题 技能 答案文件；--json 输出结构化结果

# 检索与评测
uv run ielts-tutor search "traffic in big cities" -k 5
uv run ielts-tutor search "over-use of linkers" --source rubric --skill writing
uv run ielts-tutor eval --show-misses
uv run ielts-tutor eval-scorer [--llm 150]                  # --llm 会产生 API 费用
uv run ielts-tutor kg-stats
```

每次练习的完整过程（检索证据、各来源分数、LLM 输出、耗时与调用次数）保存在 `var/sessions/<id>.json`，学生档案在 `var/learners/`。一次带 LLM 的写作练习约 3–4 次调用，口语约 3 次。

## 代码结构

| 路径 | 内容 |
| --- | --- |
| `data/ielts/kg/*.yaml` | 种子本体：技能、评分维度与 band 描述、题型、写作结构、话题、语法点、错误类型、听力主题 |
| `data/ielts/corpus/*.jsonl` | 语料 RAG 数据（`CorpusChunk`）；示例为自编练习题 |
| `data/ielts/glossary.yaml` | 雅思术语别名表（含中文），用于扩展关键词检索的查询 |
| `data/ielts/eval/retrieval_queries.yaml` | 检索评测集（40 条，含 3 条中文） |
| `data/ielts/scoring/*.jsonl` | 评分 RAG 的已评分样本（`ScoredSample`），用 `import-scored` 生成 |
| `data/ielts/corpus/private/`、`data/ielts/scoring/private/` | 有版权的官方材料（ielts.org 样题、剑桥真题），只在本地，不进 git |
| `data/ielts/answers/private/`、`data/ielts/explanations/private/` | 剑桥听力/阅读答案、中文解析分段（同样只在本地） |
| `src/ielts_tutor/kg/` | 图谱 schema（边的类型约束）、YAML 加载与校验、NetworkX 存储、导出 Cypher（Neo4j）、LLM 打标签 |
| `src/ielts_tutor/rag/` | 嵌入、BM25 + 向量混合索引、三路检索器、RRF 融合、建索引 |
| `src/ielts_tutor/llm/` | 可切换 LLM：Anthropic SDK / OpenAI 兼容接口 / 测试用 Fake |
| `src/ielts_tutor/importers/` | 各数据源导入：剑桥真题、答案页、中文解析 OCR、ielts.org |
| `src/ielts_tutor/agents/` | M2 多 Agent 写作闭环：规划、出题、作答分析、评分、反馈、学生建模，LangGraph 编排 |
| `src/ielts_tutor/writing/` | 写作特征、评分模型（特征 + 嵌入 + 梯度提升）、LLM 考官、评估与融合权重 |
| `src/ielts_tutor/speech/` | 本机语音识别（faster-whisper，词级时间戳）与流利度特征 |
| `src/ielts_tutor/web/` | FastAPI 接口 + 单页网页 |
| `src/ielts_tutor/grading/` | 客观题判分（答案匹配、原始分换算分数段）；写作/口语评分在后续里程碑 |

## 开发

```bash
uv run pytest              # 66 项，全部用 FakeLLM，不需要 API key 和真题
```

- 分支：开发在 `m1-kg-rag`，阶段完成后合并到 `main`。
- 不要提交：`.env`、`var/`、任何 `data/*/*/private/` 下的内容和真题 PDF/音频（已在 `.gitignore`）。
- macOS 13：`pyproject.toml` 已把 torch、pymupdf、onnxruntime、av 限制在仍提供 macOS 13 安装包的版本。

## 已知局限与下一步

| 问题 | 影响 | 计划 |
| --- | --- | --- |
| 写作训练数据的分数像是机器生成的 | 模型 QWK 只有 0.43 | 换用或自建可信的人工标注；用真考官样本校准；GPU 上微调 DeBERTa |
| Task 1 没有训练数据 | 只靠 LLM，一律待复核 | 收集 Task 1 评分样本 |
| Whisper 会修正口误、丢填充词 | 口语 GRA、FC 可能偏高 | 用真人录音评估，换 `small.en`，试逐字转写方案 |
| 发音只看识别置信度 | P 分基本没有区分度 | 接入 GOP / MFA 发音评估，自建口语标注集 |
| 剑14–20 多为扫描版 | 题库少了一半 | OCR 导入 |
| GT 阅读没有页码清单 | 网页上做不了 GT 阅读 | 补清单 |
| 语料话题标签不全 | 选题匹配不够准 | `tag-corpus`（约 500 次 LLM 调用） |
| 掌握度只是滑动平均 | 推题不够个性化 | M4：BKT + 学习报告 |

路线图和优先级见 [docs/PRODUCT.md](docs/PRODUCT.md#8-路线图)。

## 技术细节

### 检索方案与评测

三路检索都用"关键词（BM25，词干化）+ 语义向量"混合，两边结果用 RRF 融合。别名表只扩展关键词那一侧的查询，语义向量那一侧用原始查询。`uv run ielts-tutor eval` 的结果（40 条查询，2026-11-18）：

| 方案 | hit@1 | hit@3 | MRR |
| --- | --- | --- | --- |
| 字面哈希 | 0.57 | 0.68 | 0.67 |
| 字面哈希 + 别名表 | 0.68 | 0.85 | 0.78 |
| bge-small-en | 0.65 | 0.88 | 0.77 |
| **bge-small-en + 别名表（默认）** | **0.80** | **1.00** | **0.89** |

评测集很小，而且查询和别名表出自同一作者，数字会偏乐观；扩充评测集时，最好收集真实学生的提问。

导入剑桥真题后（语料从 15 条增加到约 1,300 个检索片段，2026-11-22），等权 RRF 下语料检索的 hit@1 从 1.00 掉到 0.40：长篇阅读和听力原文在关键词一侧占了优势。现在关键词一侧按来源加权（`rag/build.py` 中的 `BM25_WEIGHTS`：语料 0.5、话题 0.5、评分 1.0；评分检索依赖别名表扩展出的关键词，所以保持 1.0）。默认方案的结果：全部 hit@1 0.82 / MRR 0.88，评分 0.94，话题 0.86，语料 0.60。语料这一项偏低，主要是因为评测集的标准答案只列了自编题，而剑桥真题里往往有同样相关的题排在前面。真题不进仓库，所以评测集里也不能引用它们的 id。

### 注意

- `01_skills_criteria.yaml` 中的 band 描述是官方 2023 年 5 月版评分描述（Band 1–9）的摘要，附原文页码。

- 换嵌入模型后必须重新 `build`（索引会记录建索引时用的模型，不一致会报错）。bge-small-en 只懂英文，中文提问目前靠别名表；需要更好的中文支持可换 `BAAI/bge-m3`。
- macOS 13 上 torch 最高只能装 2.11（2.12 起只提供 macOS 14+ 的安装包），`pyproject.toml` 已做限制。
- 真题语料版权归各方所有，只作内部检索，不要提交到仓库。

### 剑桥真题导入（`import-cambridge`）

只解析**有文字层**的 PDF；扫描版会被跳过并提示需要先 OCR。每套题拆成：听力原文（按 Section/Part，自动关联对应 mp3）、阅读文章（含题目）、写作 Task 1/2 题目（按关键词规则标注题型）、口语 Part 1–3，以及考官范文。书后的"Sample answers"是**考生作文 + 考官分数 + 考官评语**，导入评分 RAG：

- 每条记录有 `quality`（能识别的英文单词占比）。低于 0.85 的语料不进索引，以免乱码污染检索。
- 旧版（剑4–13）的考生作文是手写稿扫描，OCR 结果基本不可读（quality 0.2–0.8），这些作文只把**考官评语**放进索引（`kind: examiner_comment`，用 `RubricRetriever.examiner_comments()` 检索），不参与相似作文检索；剑17 的作文是排版文字，作文和评语都可用。
- PDF 文字缓存在 `var/cache/cambridge/`，重跑很快。

### 听力/阅读判分（`check`）

`import-cambridge` 会同时提取书后的听力和阅读答案。旧书的答案页是两三栏排版，文字层本身也是 OCR，所以要用单词坐标重建分栏，把题号和答案配对；文字层乱码或有题缺失时，再用 tesseract 重新识别这些页，两种结果按题合并。题号被 OCR 认错的（如 37 认成 31），按前后题号修正；题号丢失的，按行距推断。

答案写法保持原样：`(the/a) pianist/piano player` 表示括号里的词可写可不写，`/` 分隔多个可接受答案；"IN EITHER ORDER" 的题组不计顺序；"BOTH REQUIRED FOR ONE MARK" 要求同时答出所有词。原始分按 IELTS 公布的参考换算表换成分数段（各套题略有差异，所以是估算）。

答案文件可以每行写成 `1 litter` 的形式，也可以每行只写答案、按顺序排列，或者用 `--answers "litter, dogs, A"` 直接写在命令里。加 `--json` 输出结构化结果，供后续 Agent 使用。

剑4–13、17 共提取 3,840 题，剑17 全部完整，旧书约 2.5% 的题仍然缺失（主要是剑10 和扫描质量差的几页）。缺答案的题判分时会标出来、不计分，分数段按已有题目等比换算。要补录或修正答案，可以在 `data/ielts/answers/private/` 下放一个 `manual.jsonl`（每行一个 `AnswerKey`）：同一道题以文件名排序靠后的为准，`manual.jsonl` 排在 `cambridge.jsonl` 之后，所以会覆盖自动提取的结果。

#### 网页上的听力和阅读练习

`import-cambridge` 还会记录每套学术类题目的听力、阅读各在 PDF 的哪几页，以及听力 4 段音频，写入 `data/ielts/tests/private/cambridge.jsonl`（剑4–13、17 共 88 组）。网页的"听力 / 阅读"页直接显示书页原图（表格、地图、流程图都保留），听力可以播放各段音频，填完 40 个答题框交卷后，逐题显示对错和中文解析，分数段记入学生档案，和写作成绩一起显示在页面右上角。服务端只读取清单里登记过的文件。General Training 的阅读还没有页码清单，暂时只能在命令行 `check`。

### 中文解析（`import-explanations`）

`【解析】` 文件夹里的解析书都是扫描版。先用 tesseract（chi_sim+eng，300dpi）逐页 OCR，结果缓存在 `var/cache/cambridge_explain/`，中断后可以接着跑；全部 OCR 一遍约 3,000 页，8 核大约 1 小时。然后按"听力 Section/Part 1–4 → 阅读 Passage 1–3 → 写作 Task 1–2 → 口语 Part 1–3"的固定顺序切分：
- 一旦出现阶段倒退（比如口语之后又出现 Part 1），就认为进入了下一套题。
- 新版书的页眉（如 "Test 1 Reading"）优先采用。
- 学术类之后、没有 Reading Passage 的套题，按 General Training A/B 处理。

判错时，在对应套题、技能和 Part 的解析里，找写有题号或正确答案（"答案为 …"）的那一段。听力约 59%、学术类阅读约 68%、GT 阅读约 56% 的题能找到解析（2026-11-27 统计）。解析是中文，英文嵌入模型用不上，所以不进检索索引，只按书、套题和题号直接查。

### M2 写作闭环

一次练习分两步：`start` 由**规划**选题型、话题和薄弱维度，**出题**找一道真题或让 LLM 写新题（新题与真题的相似度超过 0.92，或缺少该题型的固定问法时重写，最多 2 次）；学生写完后 `submit` 依次运行：

1. **作答分析**：词数、段落、句长、MTLD、连接词等特征；LLM 找出语言错误，只保留原文里确实存在、错误类型是图谱节点的那些。
2. **评分**：评分模型和 LLM 考官各占一半合成四项分（Task 2；不用 LLM 时只用模型），总分按雅思规则取整。相似作文（k-NN）仍会检索出来当证据，但不计入分数。LLM 考官会拿到模型分附近几档的官方评分描述、相似作文和考官评语，每项给出理由和证据 id。LLM 与模型相差超过 1 分时，退回作答分析复查一次；仍然不一致就标记"待人工复核"。Task 1 还没有训练数据，一律标记待复核。
3. **反馈**：针对下一档评分描述、考官评语和范文给出 3–6 条建议，每条必须引用本次检索到的证据，否则丢弃。
4. **学生建模**：更新各评分维度、题型、话题和语法点的掌握度（滑动平均，M4 换成 BKT），下次规划据此选题。

每次会话的完整过程（检索证据、各来源分数、LLM 输出、各步耗时与调用次数）保存在 `var/sessions/<id>.json`，学生档案在 `var/learners/`。一次带 LLM 的练习约 4 次调用（用真题时 3 次，触发复查时再加 2 次）。

#### 评分模型的现状（2026-11-23）

轻量模型：手工特征 + bge-small 作文向量 + 题文相似度，四项分各一个梯度提升回归（Ridge 明显更差），训练集是 btnotpt 的 5,133 篇 Task 2，在留出的 908 篇上：

| | TR | CC | LR | GRA | 总分 |
| --- | --- | --- | --- | --- | --- |
| QWK | 0.50 | 0.45 | 0.28 | 0.27 | 0.43 |
| 相差 ≤ 0.5 分 | 67% | 71% | 72% | 79% | 78% |

离"可用"还有距离，主要瓶颈是**训练数据的标签不可信**：728 篇 8 分作文的四项分全是 8/8/8/8，5.5–6.5 分只集中在几种固定组合，抽看的"8 分"作文语言明显不到 8 分，像是批量生成的分数。因此：

- k-NN 先验几乎没用（QWK 约 0.07）：语义相似的作文只是话题相近。
- LLM 考官（gpt-5.5，抽 149 篇）与这些标签的 QWK 只有 0.2–0.34，按它拟合出的融合权重几乎不给 LLM 分量（存在 `var/models/fusion_fitted.json`，仅供参考；加 `--apply-weights` 才会启用）；但对照 12 篇**真考官**评分的作文（剑17、ielts.org），LLM 9/12 篇相差在 0.5 分以内，平均差 0.62；模型 4/6 篇，平均差 0.67，两者都普遍偏低。所以默认权重定为模型 0.5、LLM 0.5（2026-11-23 决定）。

改进方向：换用或自建可信的人工标注数据（这是最关键的一步）；用真考官样本校准整体偏差；在 GPU 上微调 DeBERTa（设计文档 6.1）。

### 口语（MVP）

口语和写作共用同一条 Agent 链：规划按 Part 选一道剑桥口语真题；学生在网页上录音（或命令行提交音频文件）；录音保存在 `var/audio/`，用 faster-whisper 在**本机**转写（默认 `base.en`，`IELTS_ASR_MODEL` 可换 `small.en` 等），不上传任何外部服务，转写稿再交给 LLM 分析与评分。

- **作答分析**：转写稿的文本特征 + 流利度特征（语速、句中/句间停顿、1 秒以上长停顿、填充词、重复、平均连续语流、识别置信度）；LLM 找语法和用词错误（忽略标点，填充词不算错）。
- **评分**：口语没有训练数据，所以 FC = 规则估计 0.3 + LLM 0.7，LR、GRA 全靠 LLM，P 只能由识别置信度的规则估计（LLM 只看到文字）。一律标记"待人工复核"。
- **反馈**：同写作，必须引用下一档的口语评分描述。

已知局限：Whisper 会"修正"口误和语法错误、丢掉部分填充词（设计文档 6.2 提到的问题，已用带填充词的提示缓解，但没解决），所以 GRA 和 FC 可能偏高；发音分只看识别置信度，基本没有区分度，需要 GOP/MFA 等发音模型和带分数的口语数据。
