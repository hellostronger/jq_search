# TreeWiki 设计文档：LLMWiki × PageIndex 融合知识库系统

> 版本：v0.1（设计稿）
> 日期：2026-10-19
> 状态：设计阶段，未开始编码
> 项目代号：`treewiki`（可改）

---

## 0. 摘要

**用 LLMWiki 的方式沉淀知识（原子化、互链的 markdown wiki，由 LLM 增量维护、持续复利），用 PageIndex 的方式检索（无向量的分层树索引 + LLM 推理遍历），两者用一条"溯源链"打通：**

```
答案 → wiki 页（编译后的知识）→ 树节点摘要 → 原始文档页（证据）
```

- wiki 解决 PageIndex「每次查询都重读原文、知识不积累」的问题；
- 树索引解决 LLMWiki「wiki 大了以后只能 grep、语义检索能力弱、无页级溯源」的问题。

---

## 1. 背景调研

### 1.1 PageIndex（VectifyAI，2025.09）

来源：博客 "Next-Generation Vectorless, Reasoning-based RAG" + 开源 SDK（<https://github.com/VectifyAI/PageIndex>）。
**注意：PageIndex 没有 arXiv 论文**（arXiv 2506.13766 是同名无关论文 LHM++，已核实）。官方引用格式即博客文章。

核心思想——两步：

1. **Index**：为每个文档生成一棵**树形索引**（模仿人类专家翻阅长报告时依赖的目录层级）：
   - 节点字段（来自其开源的 `*_structure.json`）：
     ```json
     {
       "title": "Financial Results for the Quarter",
       "start_index": 1,          // 起始页
       "end_index": 1,            // 结束页
       "nodes": [ /* 子节点，递归 */ ],
       "node_id": "0001",
       "summary": "LLM 生成的节点内容摘要"
     }
     ```
   - 文档级字段：`doc_name` / `doc_description` / `structure[]`
   - 两种构建方式：
     - **Flash**（默认，文本型 PDF）：从书签/标题版面直接抽取目录骨架（**不用 LLM**），LLM 只负责节点命名与摘要。快且便宜，实测约 **$0.001/页**，1000 页教科书约 1 美元、几分钟；
     - **Classic**：无结构时由 LLM 分批读页自建树。
2. **Retrieve**：LLM agent **自顶向下推理遍历**这棵树——看到当前层各子节点的标题+摘要 → 判断进入哪个分支 → 逐层展开到叶子 → 读叶子覆盖的原始页 → 生成带**页级引用**的答案。

关键性质：
- **无向量库、无切块**：相似 ≠ 相关，检索要的是相关性，相关性靠推理；
- 检索**可解释、可溯源**（能指出答案来自第几页），可跟随文档内交叉引用（"见附录 G"）；
- **上下文感知**：对话历史、领域知识可注入查询；
- 只读推理到达的节点 → 查询成本不随文档长度增长（805 页文档直接塞上下文已不可行，PageIndex 成本不变）；
- 双模型策略：**index 用廉价模型即可**（树骨架来自版面，LLM 只做摘要润色），**chat/retrieve 用最强模型**；
- 官方 benchmark：FinanceBench 98.7%（对比向量 RAG ~50%）。

### 1.2 LLM Wiki（Karpathy，2026.04）

来源：Karpathy 的 gist "LLM Wiki"（<https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>），权威一手描述。

核心思想：**不在查询时从原文检索，而是让 LLM 增量构建并维护一个持久 wiki**——介于人与原始来源之间的结构化互链 markdown 集合。知识被"**编译一次、持续保鲜**"，而不是每次查询重新推导。wiki 是**复利资产**：交叉引用已建好、矛盾已标注、综合已形成，每加一个来源、问一个问题都在变富。

三层架构：
- **Raw sources**（`raw/`）：不可变原始来源，LLM 只读不写；
- **The wiki**（`wiki/`）：LLM 全权维护的 markdown（来源摘要页、实体页、概念页、对比页、综述页）；
- **The schema**（`CLAUDE.md` / `AGENTS.md`）：约定 wiki 结构、命名、各操作流程的配置文档，与 LLM 共同演化。

三个操作：
- **Ingest**：读新来源 → 写来源摘要页 → 更新受影响的实体/概念页（一个来源可能触及 **10–15 页**）→ 更新 `index.md` → 追加 `log.md`；
- **Query**：基于 wiki 检索、阅读、综合，带引用；**好答案回填为新 wiki 页**（探索也复利）；
- **Lint**：健康检查——跨页矛盾、被新来源取代的过时断言、孤儿页、被提及但缺页的概念、缺失交叉引用、数据缺口。

两个特殊文件：
- **`index.md`**（内容向）：全 wiki 目录，按类别组织，每页一行「链接 + 一句话摘要」。查询时先读 index 再钻取。~100 来源 / 数百页规模下够用，**避免 embedding RAG 基础设施**；
- **`log.md`**（时间向）：追加式日志，条目统一前缀（`## [2026-04-02] ingest | Article Title`），可用 `grep "^## \[" log.md | tail -5` 解析。

规模上限时的建议：引入本地 markdown 搜索引擎（如 qmd：BM25+向量+重排，有 CLI 和 MCP），而非一开始就上向量库。

### 1.3 互补性分析（为什么值得融合）

| 维度 | LLMWiki | PageIndex |
|---|---|---|
| 知识形态 | **跨文档**蒸馏、综合、互链（编译产物） | **单文档**内导航（原始结构） |
| 时间维度 | 复利、持续维护 | 每文档一次性构建 |
| 检索方式 | index.md + grep（字面向） | 推理遍历树（语义向） |
| 溯源 | 弱（引用到来源粒度） | 强（页级） |
| 规模瓶颈 | wiki 变大后 index+grep 不够 | 跨文档综合、知识积累弱 |

痛点正好配对：
- LLMWiki 的检索短板（grep 找不到"相关但不相似"的内容）**恰是** PageIndex 的强项 → **给 wiki 建树**；
- PageIndex 的综合短板（无积累、每问重读原文）**恰是** LLMWiki 的强项 → **wiki 作编译层**；
- LLMWiki 蒸馏长文档时需要可靠的阅读路径 → **PageIndex 树正好是受控的阅读骨架**（蒸馏沿树读摘要+必要页，而非全文塞上下文）。

### 1.4 相关工作定位

- **RAPTOR**：对切块做层次摘要树，但检索仍靠 embedding——本系统检索靠推理遍历，且知识有复利层；
- **GraphRAG**：跨文档实体图谱，构建重、查询贵——本系统以「树 + [[互链]]」为主，图退化为 wiki 内链接；
- **qmd**：markdown 混合检索引擎——可作为 v2 可选的兜底后端，v1 不引入向量。

---

## 2. 总体设计

### 2.1 核心概念

1. **双层索引**
   - **源树（source tree）**：每个原始文档一棵 PageIndex 式树，随文档不可变；
   - **wiki 树（wiki tree）**：对整个 wiki 建的目录树（领域 → 主题簇 → 页面），随 ingest 增量维护。
2. **溯源链**：`answer → wiki page → citation(doc_id, pages) → tree node → raw page text`。任何结论都能追到原始证据页。
3. **快慢双路**：query 默认走 wiki（快、已综合）；当问题粒度超出 wiki 内容 / 数据陈旧 / 明确要求原始细节时，**下钻**源树（准、有据）。
4. **复利循环**：ingest 沉淀 wiki；query 的好答案回填 wiki；lint 保持健康；update 做外科手术式同步。

### 2.2 存储布局

```
kb/
├── raw/                               # 不可变原始来源（pdf/doc/docx/html/图片走 MinerU；md/txt 本地解析）
├── cache/                              # 可整体 gitignore
│   ├── parse/{doc_id}/                # 解析输出：document.md / content_list.json（MinerU 云结果或本地文本分页）
│   └── pages/{doc_id}/p001.md ...     # 按 page_idx 重组的页级缓存（树索引与 read_pages 的数据源）
├── index/
│   ├── sources.json                   # 来源注册表
│   ├── trees/{doc_id}.json            # 源树（PageIndex 兼容格式）
│   └── wiki_tree.json                 # wiki 树
├── wiki/
│   ├── entities/                      # 实体页（公司、人、产品…）
│   ├── topics/                        # 概念/主题页
│   ├── sources/                       # 来源摘要页（每来源一页）
│   ├── analyses/                      # query 回填的分析页
│   ├── index.md                       # 内容目录（Karpathy 式）
│   ├── log.md                         # 追加式时间线
│   └── SCHEMA.md                      # wiki 约定（人生读 + agent 读）
├── log/agent_runs/*.jsonl             # 每次 agent 运行轨迹（调试 + 成本核算）
└── config.yaml
```

整个 kb 就是一个 git 仓库：版本历史、diff、回滚全部免费（Karpathy 原话：The wiki is just a git repo of markdown files）。

### 2.3 核心数据模型

**来源注册表 `sources.json`**：

```json
{
  "src_0007": {
    "path": "raw/q1-fy25-earnings.pdf",
    "sha256": "…",
    "pages": 2,
    "description": "v1.0 平台部署文档",   // 上传时的用户补充说明（可空）
    "ingested_at": "2026-10-19T14:03:00+08:00",
    "version": 1,
    "status": "active"          // active | superseded
  }
}
```

**源树 `index/trees/src_0007.json`**（与 PageIndex 结构兼容）：

```json
{
  "doc_id": "src_0007",
  "doc_name": "q1-fy25-earnings.pdf",
  "doc_description": "A comprehensive financial report detailing …",
  "user_description": "v1.0 平台部署文档",
  "structure": [
    {
      "title": "Financial Results for the Quarter",
      "start_index": 1, "end_index": 1,
      "nodes": [ /* 递归 */ ],
      "node_id": "0001",
      "summary": "…"
    }
  ]
}
```

**wiki 页面**（markdown + YAML frontmatter，Obsidian 兼容）：

```markdown
---
id: ent_disney
title: 华特迪士尼公司
type: entity                # entity | topic | source | analysis
tags: [媒体, 流媒体]
sources: [src_0007, src_0003]
citations:                  # 溯源链中段：claim 级
  - doc: src_0007
    pages: [1, 2]
    claim: "FY25 Q1 收入 $24.7B，同比 +5%"
updated: 2026-10-19
stale: false
---
正文。互链写法：[[ent_disney_plus]]、[[src_0007]]。
```

**wiki 树 `index/wiki_tree.json`**：

```json
{
  "built_at": "2026-10-19T15:00:00+08:00",
  "page_count": 87,
  "root": {
    "title": "知识库",
    "summary": "覆盖 3 个领域的个人知识库…",
    "children": [
      {
        "title": "公司实体",
        "summary": "各公司财务与经营状况…",
        "pages": ["ent_disney", "ent_netflix"],
        "children": [ /* 更细的主题簇 */ ]
      }
    ]
  }
}
```

### 2.4 四条流水线

#### Ingest（吸收一个来源）

```
raw file + 可选补充描述（版本/用途，写入来源与树元数据，并提示文档描述生成）
  │ 1. parse（PDF/DOC/DOCX/HTML/图片 → MinerU：版面分析、自动 OCR、表格 HTML、公式 LaTeX，
  │    输出 markdown + 按 page_idx（0 基）重组成页级缓存；
  │    md/txt 不走 MinerU：直接读文本，按标题与长度切合成页）
  ▼
cache/parse/{doc_id}/ + cache/pages/{doc_id}/
  │ 2. tree build：优先 flash（MinerU 识别的标题层级 / PDF 书签 → 骨架，LLM 仅命名+摘要）
  │    无结构时 classic（LLM 分批读页建树）；产出 index/trees/{doc_id}.json
  ▼
source tree
  │ 3. distill：agent 沿树深度优先阅读（先读全部节点摘要，
  │    仅对信息密集叶节点精读原始页）→ 产出结构化提取：
  │    entities[] / topics[] / claims[]（每条 claim 带 doc + pages）
  ▼
extraction
  │ 4. wiki write（外科手术式）：
  │    新建/更新实体页、概念页、来源摘要页；
  │    与既有内容矛盾时显式标注「⚠️ 冲突：旧说 X（src_0003），新说 Y（src_0007）」；
  │    更新 index.md，追加 log.md（## [2026-10-19] ingest | q1-fy25-earnings.pdf）
  ▼
  │ 5. wiki tree update：新页插入最合适路径（v1 可全量重建，页数少时够快）
```

#### Query（问答）

```
用户问题
  │ 1. （可选）改写/分解：多跳问题拆成子问题
  ▼
  │ 2. wiki 树遍历：agent 读当前层 children 的 title+summary
  │    → 选择展开哪些分支（可多选、可回溯、有步数预算）
  ▼
  │ 3. 读 wiki 页；沿 [[互链]] 跳转（≤3 跳）；grep_wiki 作字面量兜底
  ▼
  │ 4. 是否下钻？（问题粒度 > wiki 内容 / stale=true / 用户要原始数据 / 低置信度）
  │    是 → 打开 cited 源树 → PageIndex 式遍历 → 读原始页
  ▼
  │ 5. 综合答案：正文双重标注引用（wiki 页 + doc:page）；
  │    全程轨迹写入 log/agent_runs/
  ▼
  │ 6.（--file-back）答案回填 analyses/，更新 index.md / log.md → 复利
```

**查询模式（`--mode`，三种）**：

| 模式 | CLI | 执行方式 | 适用场景 | 相对成本 |
|---|---|---|---|---|
| **先后（级联，默认）** | `--mode cascade` | wiki 优先；判定不足（粒度 / stale / 低置信度）再下钻源树 | 日常问答 | 低 |
| **独立（单路）** | `--mode wiki` / `--mode raw` | 只走一条路，互不交叉 | `wiki`：浏览概览、确认 wiki 新鲜时；`raw`：要原始精度、wiki 疑似过时、与 wiki 路 A/B 评测 | 最低 |
| **并行（交叉验证）** | `--mode parallel` | 两路异步并行，汇总层对比答案 | 高价值 / 审计类问题 | ≈2× |

并行模式汇总规则：两路一致 → 输出并标注「双路验证一致」；不一致 → 以 raw 为事实基准输出、附冲突说明，并把矛盾登记进 lint 冲突队列——**并行模式因此兼作 wiki 质量的反馈环**（发现的陈旧/蒸馏错误直接触发修复）。模式可按次指定，也可在 config 设默认（`query.mode`）。

#### Update（同一来源出新版本）

1. re-parse + rebuild tree；
2. **树 diff**：以页范围为锚对齐新旧节点，得到 added / removed / changed 节点集合；
3. 只对 changed 节点相关的 claims 触发 wiki 外科手术式更新（树 diff 让「surgical update」免费）；
4. 旧版本 `status: superseded`，wiki 中受影响引用更新。

#### Lint（定期体检）

- **矛盾检测**：跨页 claims 抽样/聚类后送 LLM 判定冲突，产出冲突清单供人裁决；
- **陈旧检测**：来源已 update 但 wiki 未同步；`stale` 标记扫描；
- **结构健康**：孤儿页（无入链）、高频被 `[[]]` 引用但缺页的概念、index.md/log.md 与实际文件一致性；
- **索引漂移**：wiki 树与实际页面集是否一致，不一致时建议重建。

### 2.5 Agent 工具面（query agent）

PageIndex 风格，但作用于两种树：

| 工具 | 作用 |
|---|---|
| `expand_wiki(node_path)` | 返回该 wiki 树节点的 children 摘要 |
| `read_wiki_page(page_id)` | 渲染 wiki 页正文 |
| `grep_wiki(pattern)` | 字面量搜索兜底（ripgrep） |
| `expand_source(doc_id, node_path)` | 返回源树节点 children 摘要 |
| `read_pages(doc_id, a, b)` | 读原始页文本 |
| `finish(answer, citations)` | 结束并结构化输出 |

预算控制：最大步数、最大 token、最大成本三重上限，超限强制收敛。

---

## 3. 技术选型

| 项 | 选择 | 理由 |
|---|---|---|
| 语言 | Python 3.11+ | LLM 生态成熟；两参考实现均为 Python |
| LLM 接入 | OpenAI 兼容 SDK：`base_url` / `api_key` / `index_model` / `chat_model` 全部可配置，运行时可在 Web 配置页修改 | 不锁定供应商：OpenAI / DeepSeek / Qwen / 本地 vLLM 均可 |
| 文档解析 | PDF/DOC/DOCX/HTML/图片走 MinerU，**默认 mineru.net 云 API**（`parse.backend: mineru_api`，免部署、解析快）；md/txt 本地直接读取、按标题与长度分页；本地 Python API 可选（本机无 CUDA 仅 CPU） | 版面分析 + OCR + 表格/公式识别，扫描件也覆盖；MinerU 输出 markdown + 页级结构化 JSON（`page_idx` 0 基，树内统一转 1 基）；注意 MinerU 的 markdown 是输出格式，其输入不接受 `.md` |
| 结构化输出 | JSON mode / `response_format` | 建树与蒸馏都走 JSON |
| Agent 循环 | 自研轻量 loop（工具调用） | 不引重型框架，行为可控可调试 |
| CLI | typer + rich | |
| Web UI | FastAPI + 轻量单页前端 | **设置页**：可视化配置 LLM（base_url/key/双模型）、`parse.backend`、mineru.token、预算等，读写 config.yaml；后续可扩展问答界面 |
| 存储 | 纯文件（markdown + json） | git 友好，符合 LLMWiki 哲学 |
| 并发 | asyncio + semaphore | 页级摘要/蒸馏并行 |
| 测试 | pytest | |

**双模型策略**（沿用 PageIndex 经验）：`index_model`（建树、蒸馏，用便宜模型）与 `chat_model`（查询遍历、综合，用最强模型）分开配置。

---

## 4. 代码结构

```
treewiki/
├── pyproject.toml
├── config.yaml               # llm.* / parse.backend(默认 mineru_api) / mineru.token / query.mode(默认 cascade) / kb 路径 / 预算
├── treewiki/
│   ├── config.py
│   ├── core/
│   │   ├── llm.py            # 客户端封装：chat / structured / 并发 / 用量统计
│   │   ├── mineru.py         # MinerU 接入：本地/云后端调度、输出解析、按 page_idx 重组页缓存
│   │   ├── tree.py           # 树模型：flash/classic 构建、校验、diff、遍历助手
│   │   ├── wiki.py           # 页面模型、frontmatter、[[links]]、index.md/log.md 读写
│   │   └── store.py          # KB 布局、doc_id 分配、sources.json、页缓存
│   ├── pipelines/
│   │   ├── ingest.py         # parse → tree → distill → wiki write → wiki tree update
│   │   ├── query.py          # wiki 优先、下钻源树、答案回填
│   │   ├── update.py         # re-parse → tree diff → surgical wiki 更新
│   │   └── lint.py
│   ├── agent/
│   │   ├── tools.py          # 工具注册（schema 化，输出给 LLM function calling）
│   │   └── runner.py         # 通用 agent loop（步数/token/成本预算 + 轨迹落盘）
│   ├── web/
│   │   ├── api.py            # FastAPI：配置读写校验、连通性测试（后续：问答接口）
│   │   └── static/           # 设置页前端（原生 HTML/JS，不引重前端框架）
│   └── cli.py                # 含 tw serve 启动 Web UI
└── tests/
```

CLI 设计：

```
tw ingest <path>                 # 吸收一个来源
tw query "问题"                  # 问答；--mode cascade|wiki|raw|parallel（默认 cascade）  --file-back  --show-trace
tw lint [--fix]                  # 体检/自动修复
tw update <doc_id|path>          # 处理来源新版本
tw status                        # 统计：来源数/页数/wiki 页数/累计成本
tw tree <doc_id> | --wiki        # 打印树
tw serve                         # 启动本地 Web UI（设置页：LLM/解析/预算配置 + 连通性测试）
```

---

## 5. 里程碑

| 里程碑 | 内容 | 验证标准 |
|---|---|---|
| **M1** 检索内核 | MinerU 解析 → 建树（flash+classic）→ 树遍历问答，页级引用；含 `tw serve` 配置页 | 用 PageIndex 仓库示例 PDF 复刻其问答效果 |
| **M2** wiki 层 | 蒸馏、页面格式、[[互链]]、index.md/log.md、来源摘要页 | ingest 一个来源，wiki 页正确生成且老页面被更新 |
| **M3** 融合查询 | wiki 树构建 + 三模式查询 agent（cascade / solo / parallel，答案回填） | 同一问题 wiki 路径显著省 token；跨文档综合题可答；parallel 模式能检出 wiki 陈旧 |
| **M4** 维护 | update（树 diff + surgical 更新）、lint | 来源换新版后仅相关 wiki 页变化 |
| **M5** 评测打磨 | 小型 QA 评测（vs BM25 基线）、CLI/文档 | 评测报告 + README |

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| 蒸馏成本随页数增长 | 以树摘要为阅读骨架、只精读相关叶节点；index 用便宜模型（PageIndex 实测 $0.001/页量级可期） |
| 蒸馏失真（wiki 幻觉） | claim 级 `doc:pages` 强制引用；lint 矛盾检测；git 历史兜底 |
| wiki 树随规模退化 | 增量插入 + 定期全量重建；lint 检测漂移 |
| 遍历延迟（多轮 LLM） | 摘要缓存、grep 字面短路、深度/步数/成本预算、并发展开 |
| MinerU 云 API 依赖（额度/限流、数据出本机） | 云为默认、本地为备选（`--backend mineru_local`，敏感文档单次切换）；解析结果落 cache/parse 便于重放；云 API 结果建议全量落盘减少重复调用 |
| 树 diff 对齐困难 | 以页范围为锚做节点匹配；无法对齐的子树退化为全量蒸馏 |

---

## 7. 待确认事项

1. **wiki 内容语言**：中文 / 英文 / 双语（影响 prompt 与 frontmatter 约定）；
2. **是否要向量兜底**（qmd/BM25 混合）：建议 v1 不加，纯推理遍历 + grep。

已确认决策：
- **MinerU 默认走 mineru.net 云 API**（2026-10-19；本机 AMD 显卡无 CUDA，本地 CPU 作为 `--backend mineru_local` 备选）；
- **LLM 统一 OpenAI 兼容规范，不锁定供应商**（base_url/api_key/双模型全部可配置，2026-10-19）；
- **交互形态：CLI + 本地 Web 配置页**（`tw serve`，FastAPI + 轻量单页，2026-10-19；问答等 UI 后续迭代）。

---

## 8. 参考资料

- PageIndex 官方博客：Next-Generation Vectorless, Reasoning-based RAG — <https://pageindex.ai/blog/pageindex-intro>
- PageIndex 开源仓库（SDK、示例树 JSON、benchmark）— <https://github.com/VectifyAI/PageIndex>
- Karpathy, "LLM Wiki" gist（一手描述）— <https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f>
- Karpathy 原始推文 — <https://x.com/karpathy/status/2039805659525644595>
- MinerU 文档解析引擎（OpenDataLab）— <https://github.com/opendatalab/MinerU> · 输出格式文档 — <https://opendatalab.github.io/MinerU/reference/output_files/> · 云 API — <https://mineru.net/apiManage/docs>
- 相关工作：RAPTOR、GraphRAG（对比见 §1.4）
