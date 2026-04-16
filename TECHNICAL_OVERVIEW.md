# Climate Policy RAG System — 技术概览

> 面试用技术文档 · 2026-03-25

---

## 一、项目背景与问题定义

**背景**：美国亚利桑那州（AZ）、新墨西哥州（NM）、俄克拉荷马州（OK）境内的美洲原住民社区面临气候变化威胁（干旱、洪灾、野火、热浪），但难以从海量政策文件中快速找到适用于自己部落、县或市的政策信息。

**我解决的核心问题**：将 206 份多层级气候政策 PDF（州级/县级/市级/部落级，总计 10,119 个文本块）转化为可自然语言查询的知识库，并生成带明确来源标注的答案。

---

## 二、系统架构总览

```
用户查询
  │
  ▼
[查询理解层]  extract_geo_entities()
  - 规则匹配：部落/县/州名词典
  - LLM 兜底：claude-haiku（仅规则无命中时触发）
  │
  ▼
[混合检索层]  hybrid_search()
  ├─ 向量检索：Qdrant local（gemini-embedding-001, 3072 dim）
  ├─ 关键词检索：BM25Okapi（rank-bm25）
  └─ RRF 融合 + retrieval_tier 加权
  │
  ▼
[重排层]  rerank()
  - cross-encoder/ms-marco-MiniLM-L-6-v2（本地推理）
  - per-document 去重（每文档最多 2 chunk）
  - cross_score < -1.0 过滤
  │
  ▼
[生成层]  generate_answer()
  - claude-sonnet-4-6 via NYU Portkey @vertexai
  - 强制 [Doc N] 溯源 + Sources 列表
  │
  ▼
Streamlit Web UI（app.py）
```

---

## 三、数据处理管道（P0）

### 3.1 文档来源
- **206 份 PDF**：州政府、县政府、市政府、部落政府发布的气候、减灾、水资源、能源政策文件
- 两条处理路径：
  - **自动 OCR chunking**（129 份）：`OCR_chunker_full_v11_6.py`
  - **人工 chunking**（77 份，排版复杂）：`manual_chunker.py`（Streamlit 工具）

### 3.2 OCR 管道
```
PDF → pypdfium2 渲染（200 DPI）→ PaddleOCR（GPU, drop_score=0.80）
    → 多栏检测（x-gap ≥ 60px 拆列）
    → 表格提取（OpenCV 格线检测 / 无边框聚类 / LLM fallback）
    → 页面分类（TOC / Cover / Figure / Table / Text）
    → 标题检测（数字编号 / 关键词 / 位置+格式推断）
    → Chunk 合并（< 200 token 小块并入上一块）
```

**修复的关键技术问题（共 19 个 bug）：**

| 问题 | 解决方案 |
|---|---|
| 多栏布局导致文字乱序 | 同 y-band 内 x 间距 ≥ 60px 的 OCR 项拆分为独立列 |
| 运行页眉误识别为章节名 | 文档级频率统计（HEADER_FREQ_FRAC=0.40），高频文本加黑名单 |
| 正文句子误识别为标题 | title_like() 函数：末尾 stop word 过滤 + 句子开头词过滤 + cap ratio ≥ 70% |
| 宽稀疏图表页产生乱码表格 | ≥ 6 列 + 空格率 > 55% + 平均内容 < 20 字符 → 回退 prose |
| TOC 中的 Annex 条目触发附录检测 | 同页 ≥ 2 行以 Annex/Appendix X: 开头 → 判定为目录页 |

### 3.3 Chunk Schema
```json
{
  "chunk_id": "...",
  "doc_title": "2019 Maricopa County CWPP Update",
  "policy_id": "sha1_hash",
  "source_url": "https://...",
  "pdf_path": "...",
  "policy_level": "County",
  "policy_type": "Wildfire Plan",
  "state_list": ["Arizona"],
  "county_list": ["Maricopa"],
  "tribe_list": [],
  "section": "B. Prevention and Loss Mitigation",
  "section_path": ["B. Prevention and Loss Mitigation"],
  "pages": "31 - 31",
  "tokens": 847,
  "text": "...",
  "is_table": false,
  "is_appendix": false,
  "primary_tag": "action_policy",
  "secondary_tags": ["implementation"],
  "policy_score": 0.92,
  "retrieval_tier": "primary"
}
```

---

## 四、LLM 分类（Chunk Classification）

**目的**：为每个 chunk 添加语义标签，降低噪声对检索的影响。

**模型**：claude-haiku-4-5（速度快，成本低，适合批量分类）

**Prompt 逻辑**：
```
给你一段气候政策文本，判断以下字段：
1. primary_tag：文字类型（8选1，见下表）
2. secondary_tags：0-2 个补充标签
3. policy_score：0-1，这段文字对政策查询的直接价值
4. retrieval_tier：
   - primary      → 能直接回答政策问题（行动条款、资金项目）
   - secondary    → 有参考价值但非核心答案（背景说明、数据表格）
   - low_priority → 辅助性内容（附录、联系信息）
   - exclude      → 无意义内容，不入向量库（乱码、空白页）
```

**8 类标签体系**：

| 标签 | 含义 |
|---|---|
| `action_policy` | 具体政策条款、行动计划、规定 |
| `funding_program` | 资金项目、申请渠道、补贴 |
| `implementation` | 实施细节、责任主体、时间表 |
| `table_data` | 结构化数据表格 |
| `background_context` | 背景说明、风险评估 |
| `reference_appendix` | 词汇表、附录、参考文献 |
| `administrative` | 联系信息、签名页、会议记录 |
| `noise` | 乱码、空白、无意义内容 |

**标签 → retrieval_tier 映射**：

| retrieval_tier | 对应标签 | RRF 加权系数 |
|---|---|---|
| `primary` | action_policy、funding_program | ×1.3 |
| `secondary` | implementation、table_data、background_context | ×1.0 |
| `low_priority` | reference_appendix、administrative | ×0.6 |
| `exclude` | noise | 不入向量库 |

retrieval_tier 是 LLM 在分类阶段写入 chunk 的固定字段，检索时直接读取并乘以对应系数。同一标签下 haiku 也会结合 policy_score 做微调（例如 background_context 中包含具体数据的段落可升为 secondary，纯泛泛叙述降为 low_priority）。

**输出字段**：`primary_tag`、`secondary_tags`（0-2个）、`policy_score`（0-1）、`retrieval_tier`（primary/secondary/low_priority/exclude）

**分类结果**（10,119 chunks）：
- table_data 34.2%，background_context 33.6%，action_policy 13.0%
- exclude 0.9%（89 条不入向量库）
- 平均 policy_score: 0.510

---

## 五、向量化与索引（P1）

### 5.1 Embedding
- **模型**：`gemini-embedding-001`（via NYU Portkey @vertexai，dim=3072）
- **入库数量**：10,030 条（跳过 89 条 exclude）
- **向量数据库**：Qdrant local mode（本地文件存储）
- **Embedding 文本格式**：`{doc_title} | {section}\n{text[:6000]}`

### 5.2 Payload 索引（Qdrant）
7 个字段建立过滤索引：`primary_tag`、`retrieval_tier`、`policy_score`、`policy_level`、`policy_type`、`is_table`、`is_appendix`

### 5.3 BM25 索引
- **库**：`rank_bm25.BM25Okapi`
- **Tokenization**：lower + `[a-z0-9]+` 正则（英文文档，轻量高效）
- **持久化**：`pickle` 序列化到本地

---

## 六、检索系统（P2）

### 6.1 地理实体提取
```python
# 规则优先：维护 tribe/county/city/state 词典（从 metadata CSV 自动构建）
tribes = _rule_match(query, TRIBE_NAMES)  # 整词匹配，长词优先

# 歧义去除：tribe 名与 county 名重叠时，tribe 优先
# 例：Navajo 同时存在于 TRIBES 和 COUNTIES → 保留 tribe，清除 county

# LLM 兜底：规则全部无命中时调用 claude-haiku
```

**地理过滤三级回退**：
```
tribe 层过滤 → 无结果 → county 层 → 无结果 → state 层 → 无结果 → 全库
+ 始终并行补充 state 级文档（确保宏观政策不遗漏）
```

### 6.2 混合检索（Hybrid Search）
```
向量检索（Qdrant query_points）
    + BM25 检索（id_whitelist 保持地理一致性）
    ↓
RRF 融合（k=60）：score = Σ 1/(60 + rank_i)
    ↓
retrieval_tier 加权：primary ×1.3，secondary ×1.0，low_priority ×0.6
```

**选择 RRF 而非线性加权的原因**：RRF 对各检索系统的分数尺度不敏感，稳健性强，无需调参。

### 6.3 Cross-encoder 重排
- **模型**：`cross-encoder/ms-marco-MiniLM-L-6-v2`（本地，~67M 参数）
- **输入**：(query, chunk.text[:512]) 文本对
- **作用**：弥补双塔模型的语义精度不足，显著提升 top-k 精准率
- **后处理**：per-document 去重（每文档最多 2 chunk）+ cross_score < -1.0 过滤

---

## 七、答案生成（P3）

### 7.1 System Prompt 设计
```
角色：服务 AZ/NM/OK 原住民社区的气候政策助手
约束：
  1. 只使用提供的 context，禁止捏造
  2. 每条关键陈述用 [Doc N] 标注
  3. 输出结构：tribal 信息优先 → county → state
  4. 末尾必须附 Sources 列表（含 URL）
  5. 无相关信息时明确告知，不硬撑答案
```

### 7.2 Context 构建
每个送入 LLM 的 chunk 携带：文档名、章节、页码、类型标签、优先级、URL、正文（截断至 1500 字符）

### 7.3 模型选择
- **生成**：`claude-sonnet-4-6`（@vertexai via Portkey）——推理能力强，溯源格式稳定
- **分类**：`claude-haiku-4-5`——批量任务，速度/成本优先
- **地理提取兜底**：`claude-haiku-4-5`——轻量 JSON 提取

---

## 八、关键设计决策与 Trade-off

| 决策 | 选择 | 原因 |
|---|---|---|
| 向量模型 | gemini-embedding-001（3072d）而非 text-embedding-3-small | NYU Portkey 网关直接可用，无需额外配置 |
| 检索融合 | RRF 而非加权求和 | 不依赖分数校准，稳健，零额外参数 |
| 重排器 | 本地 cross-encoder 而非 LLM reranking | 延迟低（< 1s），无 API 成本，效果经过 MS MARCO 验证 |
| 地理过滤 | 自动三级回退而非严格过滤 | 原住民政策高度地理嵌套，严格过滤会漏掉州级关键政策 |
| Chunk 分类 | 8 类细粒度 + policy_score 而非简单二元过滤 | 支持后续加权检索和 reranker 初筛，更灵活 |
| OCR | PaddleOCR GPU 而非 Tesseract/AWS Textract | 免费，支持 GPU 加速，对政府文件体裁适配好 |

---

## 九、数据规模与性能

| 指标 | 数值 |
|---|---|
| 处理文档数 | 206 份 PDF |
| 总 Chunks | 10,119 |
| 入库向量数 | 10,030（去除 89 条 noise）|
| 向量维度 | 3,072（gemini-embedding-001）|
| BM25 语料规模 | 10,030 条 |
| 单次检索延迟（向量+BM25+RRF） | ~1-2s |
| Cross-encoder 重排延迟（top-50）| ~1-3s |
| 答案生成延迟 | ~3-8s |
| Qdrant 本地存储大小 | ~500MB |

---

## 十、文件结构

```
policy_data/
├── OCR_chunker_full_v11_6.py   # OCR + Chunking 主脚本
├── manual_chunker.py            # Streamlit 人工分块工具
├── classify_chunks.py           # LLM 批量分类（Haiku）
├── embed_chunks.py              # Embedding + Qdrant 入库
├── build_bm25.py                # BM25 索引构建
├── retriever.py                 # 混合检索模块（公开 search() 接口）
├── generator.py                 # 答案生成模块
├── app.py                       # Streamlit Web UI
├── qdrant_storage/              # Qdrant 本地向量库
├── bm25_index.pkl               # BM25 索引
├── bm25_corpus_ids.pkl          # BM25 语料 ID 映射
└── chunking_output/
    ├── all_chunks_classified.jsonl   # 全量分类后 chunks
    └── script_chunking_v11_6.jsonl   # OCR 自动 chunking 输出
```

---

## 十一、可能的改进方向（面试加分点）

1. **查询扩展**：用 LLM 将用户查询改写为多个子查询，提升 recall
2. **HyDE**（Hypothetical Document Embeddings）：先让 LLM 生成假设性答案，再用答案向量检索
3. **评估体系**：构建 20-50 条 (query, expected_doc) 对，计算 Recall@5、MRR、NDCG
4. **流式输出**：Streamlit 支持 `st.write_stream()`，可实现打字机效果
5. **部落别名扩展**：当前词典仅覆盖主要部落，可用 NER 模型自动扩充
6. **多轮对话**：保持 conversation history，支持追问（"tell me more about the funding"）
