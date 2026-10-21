"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  sources: [],
  currentDoc: null,
  tree: null,
  treeSeq: 0,
  nodeInfo: new Map(),
  rowState: new Map(),
  selectedNodeId: null,
  wiki: null,
  currentWikiPage: null,
  config: null,
  pollTimer: null,
  searchTimer: null,
  wikiPageButtons: new Map(),
  toastTimer: null,
};

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function loading(text) {
  return h("div", { class: "loading" }, h("span", { class: "spinner" }), text);
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  let data = {};
  try {
    data = await res.json();
  } catch {
    data = {};
  }
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`);
  return data;
}

function fmtNum(value) {
  return Number(value || 0).toLocaleString("zh-CN");
}

function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function pageRange(start, end) {
  return start === end ? `p${start}` : `p${start}–${end}`;
}

function countTree(nodes) {
  let count = 0;
  let leaves = 0;
  const walk = (items) => {
    for (const node of items) {
      count += 1;
      if (node.nodes && node.nodes.length) walk(node.nodes);
      else leaves += 1;
    }
  };
  walk(nodes || []);
  return { count, leaves };
}

function toast(text) {
  const box = $("toast");
  box.textContent = text;
  box.classList.remove("hidden");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => box.classList.add("hidden"), 2600);
}

function setMsg(el, text, isError = false) {
  el.textContent = text || "";
  el.style.color = isError && text ? "var(--danger)" : "";
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === name);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("active", view.id === `view-${name}`);
  });
  if (name === "source" && state.currentDoc && state.tree?.doc_id !== state.currentDoc) {
    loadTree(state.currentDoc);
  }
  if (name === "wiki" && state.wiki) renderWiki();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function syncSelects() {
  const doc = state.currentDoc || "";
  if ($("source_select").value !== doc) $("source_select").value = doc;
  if ($("query_doc").value !== doc) $("query_doc").value = doc;
}

function updateTopStatus() {
  const parts = ["M1"];
  parts.push(state.sources.length ? `${state.sources.length} 棵源树` : "等待摄取");
  if (!state.wiki) parts.push("Wiki 读取中");
  else if (state.wiki.status === "ready") parts.push(`Wiki ${state.wiki.counts.pages} 页`);
  else parts.push("Wiki 待接入");
  $("top_status_text").textContent = parts.join(" · ");
}

/* ------------------------------------------------------------------ 配置 */

async function loadConfig() {
  try {
    const data = await api("/api/config");
    state.config = data.config;
    $("llm_base_url").value = data.config.llm.base_url || "";
    $("llm_api_key").value = data.config.llm.api_key || "";
    $("llm_index_model").value = data.config.llm.index_model || "";
    $("llm_chat_model").value = data.config.llm.chat_model || "";
    $("mineru_token").value = data.config.mineru.token || "";
    $("mineru_model_version").value = data.config.mineru.model_version || "pipeline";
    $("kb_home_label").textContent = `知识库：${data.home}`;
  } catch (e) {
    setMsg($("config_msg"), `读取配置失败：${e.message}`, true);
  }
}

async function saveConfig() {
  const msg = $("config_msg");
  setMsg(msg, "保存中…");
  const payload = {
    llm: {
      base_url: $("llm_base_url").value.trim(),
      api_key: $("llm_api_key").value.trim(),
      index_model: $("llm_index_model").value.trim(),
      chat_model: $("llm_chat_model").value.trim(),
    },
    mineru: {
      token: $("mineru_token").value.trim(),
      model_version: $("mineru_model_version").value,
    },
  };
  try {
    await api("/api/config", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    setMsg(msg, "已保存到 kb/config.yaml");
  } catch (e) {
    setMsg(msg, `保存失败：${e.message}`, true);
  }
}

async function testLLM() {
  const btn = $("btn_test");
  const msg = $("config_msg");
  btn.disabled = true;
  setMsg(msg, "测试中…");
  try {
    const r = await api("/api/test-llm", { method: "POST" });
    if (r.ok) setMsg(msg, `LLM 连通（${r.elapsed}s）：${r.reply}`);
    else setMsg(msg, `LLM 不通：${r.error}`, true);
  } catch (e) {
    setMsg(msg, `请求失败：${e.message}`, true);
  }
  btn.disabled = false;
}

/* ------------------------------------------------------------------ 来源 */

async function loadSources() {
  try {
    const data = await api("/api/sources");
    state.sources = data.sources || [];
    if (!state.currentDoc || !state.sources.some((s) => s.doc_id === state.currentDoc)) {
      state.currentDoc = state.sources[0]?.doc_id || null;
      state.tree = null;
    }
    renderSourceSelects();
    renderSourcesTable();
    renderSourceMeta();
    renderOverview();
    updateTopStatus();
    if (state.tab === "source" && state.currentDoc) await loadTree(state.currentDoc);
  } catch (e) {
    toast(`读取来源失败：${e.message}`);
  }
}

function renderSourceSelects() {
  for (const id of ["source_select", "query_doc"]) {
    const select = $(id);
    const prev = state.currentDoc || select.value;
    select.replaceChildren();
    if (!state.sources.length) {
      select.append(h("option", { value: "" }, "暂无文档"));
      continue;
    }
    for (const source of state.sources) {
      select.append(h("option", { value: source.doc_id }, `${source.doc_id} · ${source.name}`));
    }
    if (prev && state.sources.some((s) => s.doc_id === prev)) select.value = prev;
  }
  syncSelects();
}

function renderSourcesTable() {
  const tbody = $("sources_table");
  tbody.replaceChildren();
  if (!state.sources.length) {
    tbody.append(h("tr", {}, h("td", { colspan: "6", class: "muted" }, "暂无来源，上传一个文档开始。")));
    return;
  }
  for (const source of state.sources) {
    const row = h(
      "tr",
      { class: `src-row${source.doc_id === state.currentDoc ? " selected" : ""}` },
      h("td", {}, h("code", {}, source.doc_id)),
      h("td", {}, source.name),
      h("td", {}, fmtNum(source.pages)),
      h("td", {}, fmtNum(source.nodes)),
      h("td", { class: "muted" }, fmtDate(source.ingested_at)),
      h("td", {}, h("span", { class: `status-chip ${source.status === "active" ? "ok" : ""}` }, source.status || "—")),
    );
    row.addEventListener("click", () => selectDoc(source.doc_id));
    tbody.append(row);
  }
}

async function selectDoc(docId, { load = true } = {}) {
  state.currentDoc = docId;
  state.selectedNodeId = null;
  syncSelects();
  renderSourcesTable();
  if (load) await loadTree(docId);
}

/* ------------------------------------------------------------------ 源树 */

function indexTree(nodes, parentId = null, trail = []) {
  for (const node of nodes) {
    const path = [...trail, node.title];
    state.nodeInfo.set(node.node_id, { node, parentId, path });
    if (node.nodes && node.nodes.length) indexTree(node.nodes, node.node_id, path);
  }
}

async function loadTree(docId) {
  if (!docId) return;
  if (state.tree?.doc_id === docId) return;
  const seq = ++state.treeSeq;
  state.tree = null;
  state.nodeInfo = new Map();
  state.rowState = new Map();
  $("tree_root").replaceChildren(loading("加载源树索引…"));
  try {
    const data = await api(`/api/tree/${encodeURIComponent(docId)}`);
    if (seq !== state.treeSeq) return;
    state.tree = data.tree;
    indexTree(state.tree.structure || []);
    renderSourceMeta();
    renderTree();
    const first = state.tree.structure?.[0];
    if (first) selectNode(first.node_id);
    else {
      $("node_detail").replaceChildren(
        h("div", { class: "empty-state compact" }, h("h3", {}, "空树"), h("p", {}, "该文档没有生成任何树节点。")),
      );
    }
  } catch (e) {
    if (seq !== state.treeSeq) return;
    $("tree_root").replaceChildren(h("div", { class: "error-box" }, e.message));
    $("source_meta").replaceChildren(h("div", { class: "error-box" }, `读取源树失败：${e.message}`));
  }
}

function renderSourceMeta() {
  const box = $("source_meta");
  box.replaceChildren();
  if (!state.currentDoc) {
    box.append(h("div", { class: "empty-inline" }, "选择一个已建树的文档，查看 PageIndex 产物。"));
    return;
  }
  const source = state.sources.find((s) => s.doc_id === state.currentDoc);
  if (!source) {
    box.append(h("div", { class: "empty-inline" }, "来源信息尚未加载。"));
    return;
  }
  const tree = state.tree?.doc_id === state.currentDoc ? state.tree : null;
  const stats = countTree(tree?.structure || []);
  const title = tree?.doc_name || source.name;
  const description = tree?.doc_description || source.doc_description || source.description || "该文档没有描述。";
  const userDesc = tree?.user_description || source.description;

  box.append(
    h(
      "div",
      { class: "source-info" },
      h(
        "div",
        {},
        h("span", { class: "eyebrow" }, "SOURCE TREE"),
        h("h3", {}, title),
        h("p", {}, description),
        h(
          "div",
          { class: "meta-badges" },
          h("span", { class: "badge" }, h("strong", {}, state.currentDoc)),
          h("span", { class: "badge" }, "页数 ", h("strong", {}, fmtNum(source.pages))),
          h("span", { class: "badge" }, "节点 ", h("strong", {}, fmtNum(stats.count))),
          h("span", { class: "badge" }, "叶节点 ", h("strong", {}, fmtNum(stats.leaves))),
          h("span", { class: "badge" }, "摄取 ", fmtDate(source.ingested_at)),
          userDesc ? h("span", { class: "badge" }, "补充描述 ", userDesc) : null,
        ),
      ),
      h(
        "div",
        { class: "meta-badges" },
        h("span", { class: "status-chip ok" }, "PageIndex 已生成"),
        h("button", { class: "btn secondary small", type: "button" }, "用它提问"),
      ),
    ),
  );
  const askBtn = box.querySelector("button");
  askBtn.addEventListener("click", () => {
    $("query_doc").value = state.currentDoc;
    switchTab("query");
    $("query_text").focus();
  });
}

function renderTree() {
  const root = $("tree_root");
  const term = $("tree_search").value.trim();
  $("tree_count").textContent = state.tree
    ? `${fmtNum(countTree(state.tree.structure).count)} 节点 · ${fmtNum(countTree(state.tree.structure).leaves)} 叶`
    : "";
  state.rowState = new Map();
  root.replaceChildren();

  if (!state.tree) {
    root.append(h("div", { class: "empty-inline" }, "暂无树索引。"));
    return;
  }
  if (term) {
    renderSearchResults(term);
    return;
  }
  const box = h("div", {});
  renderTreeLevel(state.tree.structure || [], box, 0);
  root.append(box);
  const first = state.tree.structure?.[0];
  if (first && first.nodes?.length) state.rowState.get(first.node_id)?.expand();
  if (state.selectedNodeId) {
    for (const [id, rs] of state.rowState) rs.row.classList.toggle("selected", id === state.selectedNodeId);
  }
}

function renderTreeLevel(nodes, container, depth) {
  for (const node of nodes) {
    const row = h("div", { class: "tree-row", style: `padding-left:${depth * 14 + 6}px` });
    const hasChildren = Boolean(node.nodes && node.nodes.length);
    const toggle = h(
      "button",
      { class: `tree-toggle${hasChildren ? "" : " placeholder"}`, type: "button", "aria-label": "展开或折叠" },
      hasChildren ? "›" : "·",
    );
    const label = h("span", { class: "tree-label" });
    if (node.summary) label.append(h("span", { class: "tree-summary-dot", title: "该节点有 LLM 摘要" }));
    label.append(h("span", { class: "tree-title", title: node.title }, node.title));
    label.append(h("span", { class: "tree-pages" }, pageRange(node.start_index, node.end_index)));
    row.append(toggle, label);
    row.addEventListener("click", () => selectNode(node.node_id));

    const childBox = h("div", { class: "tree-children hidden" });
    let built = false;
    let expanded = false;
    const expand = () => {
      if (!hasChildren) return;
      if (!built) {
        renderTreeLevel(node.nodes, childBox, depth + 1);
        built = true;
      }
      expanded = true;
      childBox.classList.remove("hidden");
      toggle.classList.add("open");
    };
    const collapse = () => {
      expanded = false;
      childBox.classList.add("hidden");
      toggle.classList.remove("open");
    };
    toggle.addEventListener("click", (e) => {
      e.stopPropagation();
      if (!hasChildren) return;
      if (expanded) collapse();
      else expand();
    });
    container.append(row, childBox);
    state.rowState.set(node.node_id, {
      row,
      childBox,
      toggle,
      expand,
      collapse,
      isExpanded: () => expanded,
    });
  }
}

function renderSearchResults(term) {
  const root = $("tree_root");
  const lower = term.toLowerCase();
  const matches = [];
  for (const info of state.nodeInfo.values()) {
    const node = info.node;
    if (
      node.title.toLowerCase().includes(lower) ||
      String(node.summary || "").toLowerCase().includes(lower)
    ) {
      matches.push(info);
    }
  }
  root.append(
    h("div", { class: "empty-inline" }, `匹配 ${fmtNum(matches.length)} 个节点${matches.length > 80 ? "，显示前 80 个" : ""}`),
  );
  if (!matches.length) return;
  for (const info of matches.slice(0, 80)) {
    const node = info.node;
    const result = h(
      "div",
      { class: "search-result" },
      h("strong", {}, node.title),
      h("span", {}, `${pageRange(node.start_index, node.end_index)} · 节点 ${node.node_id}`),
      node.summary ? h("p", {}, node.summary) : null,
    );
    result.addEventListener("click", () => locateNode(node.node_id));
    root.append(result);
  }
}

function locateNode(nodeId) {
  const info = state.nodeInfo.get(nodeId);
  if (!info) {
    toast(`节点 ${nodeId} 不在当前树中`);
    return;
  }
  if ($("tree_search").value.trim()) {
    $("tree_search").value = "";
    renderTree();
  }
  const chain = [];
  let cursor = info;
  while (cursor) {
    chain.unshift(cursor.node.node_id);
    cursor = cursor.parentId ? state.nodeInfo.get(cursor.parentId) : null;
  }
  for (const id of chain.slice(0, -1)) {
    const rs = state.rowState.get(id);
    if (rs && !rs.isExpanded()) rs.expand();
  }
  selectNode(nodeId);
  const rs = state.rowState.get(nodeId);
  if (rs) rs.row.scrollIntoView({ block: "center", behavior: "smooth" });
}

function selectNode(nodeId, { loadPages = false } = {}) {
  const info = state.nodeInfo.get(nodeId);
  if (!info) return;
  state.selectedNodeId = nodeId;
  for (const [id, rs] of state.rowState) rs.row.classList.toggle("selected", id === nodeId);
  renderNodeDetail(info, { loadPages });
}

function renderNodeDetail(info, { loadPages = false } = {}) {
  const box = $("node_detail");
  const { node, path } = info;
  const hasChildren = Boolean(node.nodes && node.nodes.length);
  const previewBox = h("div", { class: "page-preview-pages" });
  const loadBtn = h("button", { class: "btn secondary small", type: "button" }, "读取原文页");

  box.replaceChildren(
    h(
      "div",
      {},
      h(
        "div",
        { class: "breadcrumb" },
        path.length > 1 ? path.slice(0, -1).join(" / ") : "文档根目录",
      ),
      h("div", { class: "node-head" }, h("h2", {}, node.title), h("span", { class: "status-chip" }, `节点 ${node.node_id}`)),
      h(
        "div",
        { class: "meta-badges" },
        h("span", { class: "badge" }, `页范围 ${pageRange(node.start_index, node.end_index)}`),
        h("span", { class: "badge" }, hasChildren ? `子节点 ${node.nodes.length}` : "叶节点"),
        node.summary ? h("span", { class: "badge" }, "含 LLM 摘要") : h("span", { class: "badge" }, "无摘要"),
      ),
      h(
        "div",
        { class: "summary-box" },
        h("h4", {}, "节点摘要 · 用于导航，引用以原文为准"),
        h("p", {}, node.summary || "该节点没有摘要。可读取原文页查看内容。"),
      ),
      h(
        "div",
        { class: "page-preview" },
        h(
          "div",
          { class: "page-preview-head" },
          h("div", {}, h("span", { class: "eyebrow" }, "RAW PAGES"), h("div", { class: "muted" }, "节点覆盖的原始页，最多预览 5 页")),
          loadBtn,
        ),
        previewBox,
      ),
    ),
  );

  loadBtn.addEventListener("click", () => loadNodePages(node, previewBox, loadBtn));
  if (loadPages) loadNodePages(node, previewBox, loadBtn);
}

async function loadNodePages(node, container, button) {
  if (!state.currentDoc) return;
  button.disabled = true;
  container.replaceChildren(loading("读取原文页…"));
  try {
    const start = node.start_index;
    const end = Math.min(node.end_index, start + 4);
    const data = await api(
      `/api/pages/${encodeURIComponent(state.currentDoc)}?start=${start}&end=${end}`,
    );
    container.replaceChildren();
    if (!data.pages?.length) {
      container.append(h("div", { class: "empty-inline" }, "该节点没有可读取的页。"));
      return;
    }
    for (const page of data.pages) {
      container.append(
        h(
          "div",
          { class: "page-item" },
          h("div", { class: "page-label" }, `第 ${page.page} 页${page.truncated ? " · 内容已截断" : ""}`),
          h("pre", {}, page.text || "（空页）"),
        ),
      );
    }
    if (node.end_index > data.end) {
      container.append(
        h(
          "div",
          { class: "note" },
          `节点覆盖 ${pageRange(node.start_index, node.end_index)}，此处预览前 ${data.pages.length} 页。`,
        ),
      );
    }
  } catch (e) {
    container.replaceChildren(h("div", { class: "error-box" }, e.message));
  } finally {
    button.disabled = false;
    button.textContent = "重新读取原文页";
  }
}

async function focusNode(docId, nodeId) {
  if (state.currentDoc !== docId || state.tree?.doc_id !== docId) {
    state.currentDoc = docId;
    syncSelects();
    await loadTree(docId);
  }
  switchTab("source");
  locateNode(nodeId);
}

/* ------------------------------------------------------------------ 上传 */

async function uploadFile() {
  const file = $("file_input").files[0];
  if (!file) {
    toast("先选择文件");
    return;
  }
  const button = $("btn_upload");
  const stages = $("ingest_stages");
  button.disabled = true;
  stages.classList.remove("hidden");
  stages.textContent = "上传中…";
  try {
    const form = new FormData();
    form.append("file", file);
    form.append("description", $("file_desc").value.trim());
    const { job_id } = await api("/api/ingest", { method: "POST", body: form });
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(async () => {
      try {
        const job = await api(`/api/jobs/${job_id}`);
        stages.textContent = (job.stages || []).join("\n");
        if (job.status === "done") {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          stages.textContent += `\n完成：doc_id=${job.doc_id}`;
          button.disabled = false;
          await loadSources();
          await loadWiki();
          if (job.doc_id) {
            state.currentDoc = job.doc_id;
            await selectDoc(job.doc_id);
            switchTab("source");
          }
          toast(`摄取完成：${job.doc_id}`);
        } else if (job.status === "error") {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          stages.textContent += `\n失败：${job.error}`;
          button.disabled = false;
        }
      } catch (e) {
        clearInterval(state.pollTimer);
        state.pollTimer = null;
        stages.textContent += `\n轮询失败：${e.message}`;
        button.disabled = false;
      }
    }, 2500);
  } catch (e) {
    stages.textContent += `\n失败：${e.message}`;
    button.disabled = false;
  }
}

/* ------------------------------------------------------------------ 问答 */

async function runQuery() {
  const docId = $("query_doc").value;
  const question = $("query_text").value.trim();
  if (!docId) {
    setMsg($("query_msg"), "没有可选文档", true);
    return;
  }
  if (!question) {
    setMsg($("query_msg"), "请输入问题", true);
    return;
  }
  const button = $("btn_query");
  button.disabled = true;
  setMsg($("query_msg"), "树遍历检索中，可能需要几十秒…");
  const resultBox = $("query_result");
  resultBox.classList.remove("hidden");
  resultBox.replaceChildren(h("div", { class: "card" }, loading("LLM 正在树上定位并作答…")));
  try {
    if (state.currentDoc !== docId || state.tree?.doc_id !== docId) {
      state.currentDoc = docId;
      syncSelects();
      await loadTree(docId);
    }
    const r = await api("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: docId, question, mode: "raw" }),
    });
    renderQueryResult(r, docId);
    $("query_intro").classList.add("hidden");
    setMsg($("query_msg"), `完成：${r.rounds.length} 轮定位，读取 ${r.read_nodes.length} 个节点 / ${r.read_pages.length} 页`);
  } catch (e) {
    resultBox.replaceChildren(h("div", { class: "error-box" }, `问答失败：${e.message}`));
    setMsg($("query_msg"), "失败", true);
  }
  button.disabled = false;
}

const CITATION_RE = /(\[(?:p|P)\s*\d+(?:\s*[-–~至]\s*[pP]?\s*\d+)?\])/g;

function renderAnswerText(text) {
  const box = renderMarkdown(text);
  box.classList.add("answer-text");
  linkifyCitations(box);
  return box;
}

function linkifyCitations(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    const text = node.nodeValue;
    CITATION_RE.lastIndex = 0;
    if (!CITATION_RE.test(text)) continue;
    CITATION_RE.lastIndex = 0;
    const frag = document.createDocumentFragment();
    let last = 0;
    let match;
    while ((match = CITATION_RE.exec(text)) !== null) {
      if (match.index > last) frag.append(document.createTextNode(text.slice(last, match.index)));
      frag.append(h("span", { class: "cite-chip" }, match[1]));
      last = CITATION_RE.lastIndex;
    }
    if (last < text.length) frag.append(document.createTextNode(text.slice(last)));
    node.replaceWith(frag);
  }
}

function nodePill(docId, nodeId) {
  const info = state.nodeInfo.get(nodeId);
  const label = info
    ? `${info.node.title} · ${pageRange(info.node.start_index, info.node.end_index)}`
    : `节点 ${nodeId}`;
  const pill = h("span", { class: "node-pill" }, label);
  if (info) {
    pill.classList.add("clickable");
    pill.title = "在源树中查看该节点";
    pill.addEventListener("click", () => focusNode(docId, nodeId));
  }
  return pill;
}

function provStep(label, value, highlight = false) {
  return h("div", { class: `prov-step${highlight ? " highlight" : ""}` }, h("small", {}, label), h("strong", {}, value));
}

function renderQueryResult(r, docId) {
  const box = $("query_result");
  const pages = r.citations?.[0]?.pages || [];
  const usage = r.usage || {};
  const citationLabel = { model: "模型返回页码", text: "从 [pX] 解析", none: "无引用" }[r.citation_source] || r.citation_source;
  const fallbackLabel = {
    none: "正常定位",
    spread_sample: "定位未命中，均匀抽样叶节点",
    summary_only: "仅依据摘要作答",
  }[r.fallback] || r.fallback;

  const citationRow = h(
    "div",
    { class: "citation-row" },
    h("span", { class: "muted" }, "引用页："),
    pages.length ? pages.map((p) => h("span", { class: "cite-chip" }, `p${p}`)) : h("span", { class: "status-chip pending" }, "无页级引用"),
  );

  const provenance = h(
    "div",
    { class: "provenance" },
    provStep("问题", "用户输入"),
    h("span", { class: "prov-arrow" }, "→"),
    provStep("树定位", `${r.rounds.length} 轮`, true),
    h("span", { class: "prov-arrow" }, "→"),
    provStep("读取节点", `${r.read_nodes.length} 个`),
    h("span", { class: "prov-arrow" }, "→"),
    provStep("原文页", `${r.read_pages.length} 页`, true),
    h("span", { class: "prov-arrow" }, "→"),
    provStep("答案", r.answer_status === "ok" ? "已生成" : "空"),
  );

  const answerCard = h(
    "article",
    { class: "card answer-card" },
    h(
      "div",
      { class: "card-head" },
      h("div", {}, h("span", { class: "eyebrow" }, "ANSWER"), h("h3", {}, "带页级引用的答案")),
      h(
        "div",
        { class: "meta-badges" },
        h("span", { class: `status-chip ${r.answer_status === "ok" ? "ok" : "pending"}` }, r.answer_status === "ok" ? "已生成" : "空答案"),
        h("span", { class: "status-chip" }, citationLabel),
      ),
    ),
    renderAnswerText(r.answer || "（无答案）"),
    citationRow,
    provenance,
    h(
      "div",
      { class: "meta-badges" },
      h("span", { class: "badge" }, fallbackLabel),
      h("span", { class: "badge" }, `LLM 调用 ${usage.calls ?? 0}`),
      h("span", { class: "badge" }, `prompt ${fmtNum(usage.prompt_tokens)} tokens`),
      h("span", { class: "badge" }, `completion ${fmtNum(usage.completion_tokens)} tokens`),
    ),
  );

  const traceCard = h("article", { class: "card" });
  traceCard.append(
    h(
      "div",
      { class: "card-head" },
      h("div", {}, h("span", { class: "eyebrow" }, "RETRIEVAL TRACE"), h("h3", {}, "定位轨迹")),
      h(
        "div",
        { class: "meta-badges" },
        h("span", { class: "badge" }, `模式 raw`),
        h("span", { class: "badge" }, `读取页 ${r.read_pages.length}`),
        h("span", { class: "badge" }, `读取节点 ${r.read_nodes.length}`),
        r.lexical?.pages?.length
          ? h("span", { class: "status-chip ok" }, `字面命中 ${r.lexical.pages.length} 页`)
          : null,
      ),
    ),
  );

  if (r.rounds?.length) {
    const timeline = h("div", { class: "timeline" });
    for (const round of r.rounds) {
      const actions = h("div", {});
      if (round.expand?.length) {
        actions.append(
          h(
            "div",
            { class: "action-line" },
            h("span", { class: "action-tag expand" }, "EXPAND"),
            round.expand.map((id) => nodePill(docId, id)),
          ),
        );
      }
      if (round.read?.length) {
        actions.append(
          h(
            "div",
            { class: "action-line" },
            h("span", { class: "action-tag read" }, "READ"),
            round.read.map((id) => nodePill(docId, id)),
          ),
        );
      }
      if (!round.expand?.length && !round.read?.length) {
        actions.append(h("div", { class: "action-line muted" }, "本轮没有有效动作"));
      }
      timeline.append(
        h(
          "div",
          { class: "round-card" },
          h(
            "div",
            { class: "round-head" },
            h("span", { class: "round-num" }, `ROUND ${round.round}`),
            h("span", { class: "round-reason" }, round.reason || "无说明"),
            round.attempts > 1 ? h("span", { class: "status-chip pending" }, `${round.attempts} 次尝试`) : null,
          ),
          actions,
        ),
      );
    }
    traceCard.append(h("div", { class: "rounds" }, h("h4", {}, "逐轮决策"), timeline));
  } else {
    traceCard.append(h("div", { class: "empty-inline" }, "本次没有产生定位轮次。"));
  }

  if (r.read_nodes?.length) {
    const list = h("div", { class: "read-nodes" }, h("h4", {}, "实际读取的节点"));
    for (const node of r.read_nodes) {
      const row = h(
        "div",
        { class: "read-node-row" },
        h("strong", {}, node.title),
        h("span", { class: "node-pill" }, node.id),
        h("span", { class: "page-range" }, pageRange(node.pages[0], node.pages[1])),
      );
      row.addEventListener("click", () => focusNode(docId, node.id));
      list.append(row);
    }
    traceCard.append(list);
  }

  const raw = { rounds: r.rounds, read_nodes: r.read_nodes, read_pages: r.read_pages, lexical: r.lexical, fallback: r.fallback, citation_source: r.citation_source, usage: r.usage };
  traceCard.append(
    h(
      "details",
      { class: "raw-details" },
      h("summary", {}, "查看原始产物 JSON"),
      h("pre", {}, JSON.stringify(raw, null, 2)),
    ),
    h(
      "div",
      { class: "note" },
      r.lexical?.pages?.length
        ? `字面命中：问题关键词 ${r.lexical.terms.map((t) => `「${t}」`).join("")} 直接命中的页已作为证据注入，不依赖模型选对分支。引用页只来自本次实际读取的原文页。`
        : "引用页只来自本次实际读取的原文页。点击节点可以回到源树查看摘要与原文；当前为 raw 单路模式。",
    ),
  );

  box.replaceChildren(answerCard, traceCard);
  box.scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ------------------------------------------------------------------ Wiki */

async function loadWiki() {
  try {
    state.wiki = await api("/api/wiki/overview");
  } catch (e) {
    state.wiki = { status: "error", error: e.message, pages: [], counts: { pages: 0, links: 0 }, sections: [], special: {} };
  }
  renderOverview();
  renderWiki();
  updateTopStatus();
}

function renderWiki() {
  if (!state.wiki) return;
  const banner = $("wiki_banner");
  banner.replaceChildren();
  if (state.wiki.status === "ready") {
    banner.classList.add("ready");
    banner.append(
      h(
        "div",
        { class: "banner-main" },
        h(
          "div",
          {},
          h("span", { class: "status-chip wiki" }, "LLMWiki 产物已就绪"),
          h("h3", {}, `${fmtNum(state.wiki.counts.pages)} 个知识页 · ${fmtNum(state.wiki.counts.links)} 条 [[互链]]`),
          h(
            "p",
            {},
            `真实文件目录：${state.wiki.root}`,
          ),
        ),
        h(
          "div",
          { class: "meta-badges" },
          Object.entries(state.wiki.counts.by_type || {}).map(([type, count]) => h("span", { class: "badge" }, `${type} ${count}`)),
          state.wiki.wiki_tree ? h("span", { class: "status-chip ok" }, "wiki_tree.json 已生成") : h("span", { class: "status-chip pending" }, "wiki_tree.json 未生成"),
        ),
      ),
    );
    $("wiki_ready").classList.remove("hidden");
    $("wiki_empty").classList.add("hidden");
    renderWikiReady();
  } else {
    banner.classList.remove("ready");
    const error = state.wiki.status === "error" ? state.wiki.error : null;
    banner.append(
      h(
        "div",
        { class: "banner-main" },
        h(
          "div",
          {},
          h("span", { class: "status-chip pending" }, "LLMWiki 尚无产物"),
          h("h3", {}, "当前知识库只有 PageIndex 源树"),
          h("p", {}, error ? `读取 Wiki 失败：${error}` : "kb/wiki 目录为空。M2 生成后将自动在此展示真实知识页，不使用演示数据。"),
        ),
        h(
          "div",
          { class: "meta-badges" },
          h("span", { class: "badge" }, "M1：PageIndex 已运行"),
          h("span", { class: "badge" }, "M2：Wiki 层待接入"),
        ),
      ),
    );
    $("wiki_ready").classList.add("hidden");
    $("wiki_empty").classList.remove("hidden");
  }
}

function renderWikiReady() {
  const wiki = state.wiki;
  $("wiki_page_count").textContent = `${fmtNum(wiki.counts.pages)} 页`;
  const inventory = $("wiki_inventory");
  inventory.replaceChildren();
  for (const section of wiki.sections || []) {
    inventory.append(
      h(
        "div",
        { class: "inventory-row" },
        h("span", {}, section.label),
        h("strong", {}, fmtNum(section.count)),
      ),
    );
  }

  const list = $("wiki_page_list");
  list.replaceChildren();
  state.wikiPageButtons = new Map();
  for (const section of wiki.sections || []) {
    const pages = wiki.pages.filter((p) => p.section === section.name);
    if (!pages.length) continue;
    list.append(h("div", { class: "wiki-group-title" }, `${section.label} · ${pages.length}`));
    for (const page of pages) {
      const button = h(
        "button",
        { class: "wiki-page-item", type: "button" },
        h("strong", {}, page.title),
        h("span", {}, `${page.id} · claims ${page.citations}${page.stale ? " · stale" : ""}`),
      );
      button.addEventListener("click", () => openWikiPage(page.path));
      state.wikiPageButtons.set(page.path, button);
      list.append(button);
    }
  }

  const special = $("wiki_special_buttons");
  special.replaceChildren();
  for (const name of ["index.md", "log.md", "SCHEMA.md"]) {
    const item = wiki.special?.[name] || { exists: false, chars: 0, content: "" };
    const button = h(
      "button",
      { class: "special-btn", type: "button", disabled: !item.exists },
      h("span", {}, name),
      h("small", {}, item.exists ? `${fmtNum(item.chars)} 字符` : "未生成"),
    );
    if (item.exists) button.addEventListener("click", () => openSpecialFile(name));
    special.append(button);
  }

  const treeBox = h("div", { class: "wiki-tree-box" });
  if (wiki.wiki_tree) {
    treeBox.append(
      h("strong", {}, "Wiki 树已生成"),
      h("div", { class: "muted" }, `构建于 ${fmtDate(wiki.wiki_tree.built_at)} · ${fmtNum(wiki.wiki_tree.page_count)} 页`),
    );
  } else {
    treeBox.append(h("strong", {}, "Wiki 树未生成"), h("div", { class: "muted" }, "M3 将用它对整个 wiki 做推理遍历。"));
  }
  $("wiki_inventory").append(treeBox);

  if (wiki.pages.length && (!state.currentWikiPage || !wiki.pages.some((p) => p.path === state.currentWikiPage.path))) {
    openWikiPage(wiki.pages[0].path);
  } else if (state.currentWikiPage) {
    markWikiPageActive(state.currentWikiPage.path);
  }
}

function markWikiPageActive(path) {
  for (const [key, button] of state.wikiPageButtons) button.classList.toggle("active", key === path);
}

async function openWikiPage(path) {
  const view = $("wiki_view");
  view.replaceChildren(loading("读取 Wiki 页…"));
  markWikiPageActive(path);
  try {
    const data = await api(`/api/wiki/page?path=${encodeURIComponent(path)}`);
    state.currentWikiPage = { path: data.path };
    renderWikiPage(data);
  } catch (e) {
    view.replaceChildren(h("div", { class: "error-box" }, e.message));
  }
}

function renderWikiPage(data) {
  const view = $("wiki_view");
  const meta = data.meta || {};
  const type = meta.type || "unknown";
  const tags = Array.isArray(meta.tags) ? meta.tags : [];
  const sources = Array.isArray(meta.sources) ? meta.sources : [];
  const citations = Array.isArray(meta.citations) ? meta.citations : [];
  const body = renderMarkdown(data.body || "", (target) => followWikiLink(target));

  const sourceChips = sources.map((docId) => {
    const chip = h("span", { class: "node-pill clickable" }, docId);
    if (state.sources.some((s) => s.doc_id === docId)) {
      chip.addEventListener("click", async () => {
        await selectDoc(docId);
        switchTab("source");
      });
    } else {
      chip.classList.remove("clickable");
    }
    return chip;
  });

  view.replaceChildren(
    h(
      "div",
      { class: "wiki-page-head" },
      h("span", { class: "eyebrow" }, `WIKI PAGE · ${type}`),
      h("h2", {}, meta.title || data.path),
      h(
        "div",
        { class: "meta-badges" },
        h("span", { class: "status-chip wiki" }, String(meta.id || data.path)),
        meta.updated ? h("span", { class: "badge" }, `更新 ${meta.updated}`) : null,
        meta.stale ? h("span", { class: "status-chip stale-badge" }, "stale：可能过时") : null,
        h("span", { class: "badge" }, `claims ${citations.length}`),
        h("span", { class: "badge" }, `sources ${sources.length}`),
        h("span", { class: "badge" }, `links ${data.links?.length || 0}`),
        ...tags.map((tag) => h("span", { class: "badge" }, `#${tag}`)),
      ),
      sources.length ? h("div", { class: "meta-badges" }, h("span", { class: "muted" }, "来源："), ...sourceChips) : null,
    ),
    body,
    h(
      "details",
      { class: "raw-details" },
      h("summary", {}, "查看 frontmatter"),
      h("pre", {}, JSON.stringify(meta, null, 2)),
    ),
    h(
      "div",
      { class: "note" },
      "Wiki 页是跨文档编译产物；每条 claim 的 doc + pages 可继续下钻到对应源树与原始页。",
    ),
  );
}

function openSpecialFile(name) {
  const item = state.wiki?.special?.[name];
  if (!item?.exists) return;
  state.currentWikiPage = null;
  for (const button of state.wikiPageButtons.values()) button.classList.remove("active");
  const view = $("wiki_view");
  const content = item.content || "";
  view.replaceChildren(
    h(
      "div",
      { class: "wiki-page-head" },
      h("span", { class: "eyebrow" }, "WIKI STRUCTURE"),
      h("h2", {}, name),
      h(
        "div",
        { class: "meta-badges" },
        h("span", { class: "status-chip wiki" }, "特殊文件"),
        h("span", { class: "badge" }, `${fmtNum(item.chars)} 字符`),
      ),
    ),
    content ? renderMarkdown(content, (target) => followWikiLink(target)) : h("div", { class: "empty-inline" }, "文件为空。"),
    item.chars > content.length ? h("div", { class: "note" }, "内容较长，仅展示前 12000 字符。") : null,
  );
}

function followWikiLink(target) {
  const pages = state.wiki?.pages || [];
  const found = pages.find(
    (p) =>
      p.id === target ||
      p.title === target ||
      p.path.split("/").pop().replace(/\.md$/, "") === target,
  );
  if (found) {
    openWikiPage(found.path);
    return;
  }
  if (state.sources.some((s) => s.doc_id === target)) {
    selectDoc(target).then(() => switchTab("source"));
    return;
  }
  toast(`互链目标 [[${target}]] 尚未生成`);
}

/* ------------------------------------------------------------------ Markdown */

const INLINE_RE = /\[\[([^\]]+)\]\]|\*\*([^*]+)\*\*|`([^`]+)`|\*([^*]+)\*/g;

function appendInline(parent, text, onLink) {
  INLINE_RE.lastIndex = 0;
  let last = 0;
  let match;
  while ((match = INLINE_RE.exec(text)) !== null) {
    if (match.index > last) parent.append(document.createTextNode(text.slice(last, match.index)));
    if (match[1] !== undefined) {
      const [targetPart, labelPart] = match[1].split("|");
      const target = targetPart.split("#")[0].trim();
      const label = (labelPart || targetPart).trim();
      const link = h("button", { class: "wiki-link", type: "button", title: `互链：${target}` }, label);
      link.addEventListener("click", () => onLink?.(target));
      parent.append(link);
    } else if (match[2] !== undefined) {
      parent.append(h("strong", {}, match[2]));
    } else if (match[3] !== undefined) {
      parent.append(h("code", {}, match[3]));
    } else if (match[4] !== undefined) {
      parent.append(h("em", {}, match[4]));
    }
    last = INLINE_RE.lastIndex;
  }
  if (last < text.length) parent.append(document.createTextNode(text.slice(last)));
}

function renderMarkdown(md, onLink) {
  const root = h("div", { class: "wiki-body" });
  const lines = String(md || "").split("\n");
  let list = null;
  let listTag = "";
  let lastItem = null;
  let inCode = false;
  let codeLines = [];

  const closeList = () => {
    if (list) {
      root.append(list);
      list = null;
      listTag = "";
      lastItem = null;
    }
  };
  const closeCode = () => {
    if (!inCode) return;
    const pre = h("pre");
    pre.append(h("code", {}, codeLines.join("\n")));
    root.append(pre);
    inCode = false;
    codeLines = [];
  };

  for (const rawLine of lines) {
    const line = rawLine.replace(/\s+$/, "");
    if (line.startsWith("```")) {
      closeList();
      if (inCode) closeCode();
      else inCode = true;
      continue;
    }
    if (inCode) {
      codeLines.push(rawLine);
      continue;
    }
    if (!line.trim()) {
      closeList();
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeList();
      const block = h(`h${Math.min(6, heading[1].length)}`);
      appendInline(block, heading[2], onLink);
      root.append(block);
      continue;
    }
    if (lastItem && /^\s{2,}\S/.test(rawLine)) {
      lastItem.append(h("br"));
      appendInline(lastItem, line.trim(), onLink);
      continue;
    }
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    if (ordered || bullet) {
      const tag = ordered ? "ol" : "ul";
      if (!list || listTag !== tag) {
        closeList();
        list = h(tag);
        listTag = tag;
      }
      const item = h("li");
      appendInline(item, ordered ? ordered[1] : bullet[1], onLink);
      list.append(item);
      lastItem = item;
      continue;
    }
    if (line.startsWith(">")) {
      closeList();
      const quote = h("blockquote");
      appendInline(quote, line.replace(/^>\s?/, ""), onLink);
      root.append(quote);
      continue;
    }
    closeList();
    const para = h("p");
    appendInline(para, line, onLink);
    root.append(para);
  }
  closeCode();
  closeList();
  return root;
}

/* ------------------------------------------------------------------ 总览 */

function renderOverview() {
  const totalNodes = state.sources.reduce((sum, s) => sum + Number(s.nodes || 0), 0);
  const totalPages = state.sources.reduce((sum, s) => sum + Number(s.pages || 0), 0);
  $("stat_sources").textContent = fmtNum(state.sources.length);
  $("stat_nodes").textContent = fmtNum(totalNodes);
  $("stat_pages").textContent = fmtNum(totalPages);
  const piChip = $("pi_status");
  piChip.textContent = state.sources.length ? "M1 已运行" : "等待摄取";
  piChip.className = `status-chip ${state.sources.length ? "ok" : "pending"}`;

  const wikiChip = $("wiki_status");
  if (!state.wiki) {
    wikiChip.textContent = "读取中";
    wikiChip.className = "status-chip";
  } else if (state.wiki.status === "ready") {
    wikiChip.textContent = "已就绪";
    wikiChip.className = "status-chip wiki";
  } else {
    wikiChip.textContent = "未生成 · M2";
    wikiChip.className = "status-chip pending";
  }

  const counts = state.wiki?.counts || { pages: 0, links: 0 };
  $("stat_wiki_pages").textContent = fmtNum(counts.pages);
  $("stat_wiki_links").textContent = fmtNum(counts.links);
  const specialCount = state.wiki ? Object.values(state.wiki.special || {}).filter((x) => x.exists).length : 0;
  $("stat_wiki_special").textContent = fmtNum(specialCount);
  renderArtifacts();
}

function artifactRow(name, layer, timing, statusText, statusKind, path) {
  return h(
    "tr",
    {},
    h("td", {}, name),
    h("td", {}, layer),
    h("td", { class: "muted" }, timing),
    h("td", {}, h("span", { class: `status-chip ${statusKind}` }, statusText)),
    h("td", {}, h("code", {}, path)),
  );
}

function renderArtifacts() {
  const tbody = $("artifact_rows");
  if (!tbody) return;
  const hasSources = state.sources.length > 0;
  const totalNodes = state.sources.reduce((sum, s) => sum + Number(s.nodes || 0), 0);
  const wikiReady = state.wiki?.status === "ready";
  const wikiPages = state.wiki?.counts?.pages || 0;
  const specialCount = state.wiki ? Object.values(state.wiki.special || {}).filter((x) => x.exists).length : 0;
  const wikiTree = Boolean(state.wiki?.wiki_tree);

  tbody.replaceChildren(
    artifactRow("源树索引", "PageIndex", "摄取时", hasSources ? `已生成 · ${state.sources.length} 篇` : "未生成", hasSources ? "ok" : "pending", "kb/index/trees/{doc_id}.json"),
    artifactRow("节点摘要", "PageIndex", "摄取时", totalNodes ? `已生成 · ${fmtNum(totalNodes)} 节点` : "未生成", totalNodes ? "ok" : "pending", "随源树 JSON 落盘"),
    artifactRow("定位轨迹", "PageIndex", "每次问答", hasSources ? "运行时生成" : "待摄取", hasSources ? "ok" : "pending", "answer.rounds"),
    artifactRow("页级引用", "PageIndex", "每次问答", hasSources ? "运行时生成" : "待摄取", hasSources ? "ok" : "pending", "answer.citations[].pages"),
    artifactRow("Wiki 知识页", "LLMWiki", "M2 摄取蒸馏", wikiReady ? `已生成 · ${fmtNum(wikiPages)} 页` : "未生成（M2）", wikiReady ? "wiki" : "pending", "kb/wiki/{entities,topics,sources,analyses}/"),
    artifactRow("目录与时间线", "LLMWiki", "M2", specialCount ? `已生成 · ${specialCount} 个` : "未生成（M2）", specialCount ? "wiki" : "pending", "kb/wiki/index.md · log.md"),
    artifactRow("Wiki 树", "LLMWiki", "M2/M3", wikiTree ? "已生成" : "未生成（M2/M3）", wikiTree ? "wiki" : "pending", "kb/index/wiki_tree.json"),
  );
}

/* ------------------------------------------------------------------ 事件 */

function bindEvents() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  document.querySelectorAll("[data-goto]").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.goto));
  });

  $("source_select").addEventListener("change", (e) => selectDoc(e.target.value));
  $("query_doc").addEventListener("change", (e) => selectDoc(e.target.value, { load: false }));

  $("tree_search").addEventListener("input", () => {
    clearTimeout(state.searchTimer);
    state.searchTimer = setTimeout(renderTree, 120);
  });

  $("btn_refresh_sources").addEventListener("click", async () => {
    await loadSources();
    await loadWiki();
    toast("已刷新");
  });
  $("btn_upload_toggle").addEventListener("click", () => {
    const panel = $("upload_panel");
    const hidden = panel.classList.toggle("hidden");
    $("btn_upload_toggle").textContent = hidden ? "上传并建树" : "收起上传";
  });
  $("file_input").addEventListener("change", () => {
    $("file_display").value = $("file_input").files[0]?.name || "";
  });
  $("btn_upload").addEventListener("click", uploadFile);
  $("btn_query").addEventListener("click", runQuery);
  $("query_text").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.isComposing) runQuery();
  });
  $("btn_save").addEventListener("click", saveConfig);
  $("btn_test").addEventListener("click", testLLM);
}

function init() {
  bindEvents();
  renderOverview();
  loadConfig();
  loadSources();
  loadWiki();
}

init();
