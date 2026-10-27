# TreeWiki

LLMWiki × PageIndex 融合知识库系统（设计见 [docs/DESIGN.md](docs/DESIGN.md)）。

当前进度：**M2 wiki 蒸馏层** —— MinerU 解析 → 树索引 → wiki 蒸馏（实体/主题/关系/来源页）→ 四种查询模式（cascade / wiki / raw / parallel）+ Web 界面。

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

tw serve                       # 打开 http://127.0.0.1:8000
```

Web 页面操作：
1. **设置**：填 LLM（OpenAI 兼容 base_url / api_key / 双模型）与 MinerU 云 API token，点「测试连接」；
2. **文档**：上传 PDF/Word（MinerU 解析）或 md/txt（本地解析），可附一句补充描述（如“v1.0 平台部署文档”）→ 自动建树并蒸馏 wiki（进度轮询）；已有来源可点「蒸馏」重建；
3. **Wiki 知识层**：浏览/筛选实体、主题、来源、分析页，查看 `[[互链]]`、关键事实与页级引用；
4. **问答**：选择模式（cascade 默认 / wiki / raw / parallel）与文档（可跨文档），答案带 `doc:p` 引用与检索轨迹。

wiki 产物：`kb/wiki/{entities,topics,sources,analyses}/*.md`、`kb/wiki/index.md`、`kb/wiki/log.md`、`kb/index/wiki_tree.json`。页面 frontmatter 含 `citations`（doc + pages + claim）与 typed `relations`（predicate + target + pages），正文用 `[[页面id]]` 互链，溯源链为 **答案 → wiki 页 → 来源页/页码 → 源树节点 → 原始页**。

## CLI

```bash
tw ingest report.pdf --desc "v1.0 平台部署文档"   # 摄取一个来源（可附描述，自动蒸馏 wiki）
tw distill src_0001                            # 对已有来源重跑 wiki 蒸馏
tw wiki                                        # 列出 wiki 页；tw wiki ent_acme 查看单页
tw query "Q1 收入是多少?" --doc src_0001        # 默认 cascade
tw query "跨文档对比 A/B" --mode wiki           # wiki / raw / parallel
tw status                                      # 来源与 wiki 统计
tw tree src_0001                               # 打印树索引
```

## 实体对齐（别名索引）

文档里同一实体常有多种写法（`iPhone Duo` / `iPhone`、`宁德时代` / `宁德时代新能源科技股份有限公司`），问题用其中一种时字面检索会漏页。`entities` 用一张加权图把它们聚成实体簇，查询时按别名组召回（组内 OR、组间 AND）。

```bash
tw entities build              # 扫描全部来源页建图（纯本地，不调 LLM）
tw entities show -n 30         # 查看实体与别名
tw entities lookup 宁德时代     # 查规范名与出现页
```

算法（`treewiki/core/entities.py` + `treewiki/core/mentions.py`）：

1. **提及抽取**：引号名称、拉丁词、中文机构后缀名走规则；中文通用名用 n-gram 词频 + 左右邻接熵做新词发现。
2. **候选生成**：首字/尾二字分块 + 包含关系，避免全量 O(n²)。
3. **边权**：字面相似 0.30、上下文 Jaccard 0.10、同页共现 0.20、包含度 0.15、前缀匹配 0.25；型号后缀（Pro/Max/Ultra…）不可链接。
4. **聚类**：确定性标签传播，按频次排序、同分按字典序，结果可复现。
5. **产出**：规范名 + 别名表 + 每实体出现页，落盘 `kb/index/entities.json`；`POST /api/entities/build`、`GET /api/entities`、`GET /api/entities/lookup` 提供同等能力。

已知边界：跨语言别名（如 `NVIDIA` / `英伟达`）不靠字面信号合并，需要补充别名词典；聚类结果只影响检索召回，不改变树索引与页级引用。

### 评测（可重放语料）

原语料丢失时，可用脱敏验证集还原一份可重放页语料。它只适合验证机制、不宜比较绝对指标（重放页平均只有几十字，字面区分度被放大）：

```bash
python -m treewiki.eval.replay --dataset eval_out/dataset_public.jsonl --out eval_out/replay_kb
TREWIKI_HOME=eval_out/replay_kb tw entities build
TREWIKI_HOME=eval_out/replay_kb tw evalrun --dataset eval_out/dataset_public.jsonl \
  --mode entity --tiers A,B,S
```

801 道 A/B/S 题上的结果（1399 页、424 页有文本、223 实体 / 23 别名）：页级命中 13.5% → 51.1%，A 档保持 100%，B 档 1.4% → 52.1%，S 档 1.7% → 43.5%；平均锚词长度 11.8 → 3.0 字。**别名扩展贡献 0**——增益全部来自「用实体短名替换整段长锚词」，且主要来自 n-gram 挖出的短语（这类"实体"本身就是原句片段）。代价是 S 档返回页的精度 0.22；规则来源的实体（55 题）则同时提升召回与精度（89% → 100%，精度 0.23 → 0.42）。结论：重放语料只能确认短锚词替换有效。

### 验证集分档与 v2

题目的 tier 由生产锚词的实际命中行为决定：A（锚词全中）、B（部分命中）、S（全不中）。另有三档派生题：

- **H 多跳**：同一篇里两个不同事实合成一问（"大额存单3个月期平均利率、银行整存整取存款3个月期平均利率分别是多少？" / "…分别升至多少？"）。只合并谓词兼容的事实——平铺指标（"是多少 / 为多少"）合成"分别是多少"，变动/阈值谓词必须逐字一致（"分别升至多少"）；"股价下跌多少"与"股价上涨多少"不会被拼成"股价分别是多少"。gold 取两跳页的并集，有原文时再补标复述了两处事实的页；判分时引用页原文含该跳数字即认可，不要求必须命中来源页；
- **M 多轮**：同一主体的两个指标，第二轮省掉主语（"那三季度营收是多少？"）。执行时把 `history` 传给 `query_raw`，`resolve_followup` 用上一轮问题的主体补全后再检索，逐轮判分、全对才算过；
- **C / D**：跨天时序聚合、未收录实体拒答。

问句必须自足：单看问句认不出主体（"9月环比走高多少？"）的题在生成时就拒绝；原始语料缺失时，用脱敏集做一次升级：

```bash
tw evalset --upgrade eval_out/dataset_public.jsonl --out eval_out/dataset_public.v2.jsonl
```

v2（1000 → 956 条）：剔除 50 道缺主体题，保留 950 条，派生 H 4 条、M 2 条。H/M 偏少是脱敏集的限制——原题里"谓词一致 + 左半句是带主体的名词短语 + 两处事实同篇"的存量本来就少，完整语料重建时会更多。实体探针在 v2 的 757 道 A/B/S/H/M 题上：页级命中 14.5% → 52.6%（`anchor_gain` 288、`alias_gain` 0）；分档看 H 0% → 75%，S 1.9% → 44.4%（M 只剩 2 题，均为 100%）。

### 开源集复核（DuReader-retrieval dev 子集）

重放页平均只有几十字，绝对命中率不可信。为了在真实页密度上复核，从 [DuReader-retrieval](https://huggingface.co/datasets/zyznull/dureader-retrieval-ranking) dev 抽 300 题（277 题有正例）：每题取全部 positive passage + 最多 30 条 hard negative，按文本去重后 9936 页、331 字/页，页序用固定种子打乱，避免 gold 页集中在低页码。

277 题的页级命中：旧字面锚词 18.1% → 实体短锚词 43.0% → 加别名组 53.4%；BM25（中文二元组）83.4%。实体路线精度 0.19、MRR 0.30，BM25 召回 52.6%、MRR 0.42。结论有三点：

- 短锚词替换在真实语料上仍然有效（+24.9pp），不是重放语料造出来的；
- 别名组这次有净增益（扫回 32 题、误伤 3 题），但多数来自"造价/造价师""糖尿病/糖尿病患者"这类字面变体，不能等同于跨语言/缩写对齐；
- 词法阶段仍明显弱于 BM25。系统真正的召回要靠树遍历与 LLM 定位，锚词只是候选入口。

这轮复核还暴露并修掉两个锚词缺陷：拉丁名内部子串（iPad 同时命中 Pad/AD/IP）和疑问词（什么/怎么）被当成实体锚词，导致组间 AND 过严；修复后同一批题从 40.4% 提到 53.4%。把实体锚词组数限制到 2 组还能到 62.8%，但会改变原验证集上的行为，暂未设为默认。

## 配置

配置文件 `kb/config.yaml`（gitignore），环境变量兜底：`TREWIKI_HOME`、`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`MINERU_API_TOKEN`。

## 测试

```bash
pytest
```
