# TreeWiki

LLMWiki × PageIndex 融合知识库系统（设计见 [docs/DESIGN.md](docs/DESIGN.md)）。

当前进度：**M1 检索内核** —— MinerU 解析 → 树索引构建 → 树遍历问答（页级引用）+ Web 界面。

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"

tw serve                       # 打开 http://127.0.0.1:8000
```

Web 页面操作：
1. **设置**：填 LLM（OpenAI 兼容 base_url / api_key / 双模型）与 MinerU 云 API token，点「测试连接」；
2. **文档**：上传 PDF/Word（MinerU 解析）或 md/txt（本地解析），可附一句补充描述（如“v1.0 平台部署文档”）→ 自动建树（进度轮询）；
3. **问答**：选文档、提问 → 答案带页级引用与定位轨迹。

## CLI

```bash
tw ingest report.pdf --desc "v1.0 平台部署文档"   # 摄取一个来源（可附描述）
tw query "Q1 收入是多少?" --doc src_0001
tw status                      # 来源统计
tw tree src_0001               # 打印树索引
```

## 配置

配置文件 `kb/config.yaml`（gitignore），环境变量兜底：`TREWIKI_HOME`、`OPENAI_BASE_URL`、`OPENAI_API_KEY`、`MINERU_API_TOKEN`。

## 测试

```bash
pytest
```
