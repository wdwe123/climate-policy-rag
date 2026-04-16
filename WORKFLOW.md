# Climate Policy RAG — System Workflow

---

## 1. Ingestion Pipeline（离线，一次性构建）

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         196 个 PDF 文件                                  │
│              AZ / NM / OK  ×  Tribe / City / County / State / Federal   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  F1  页面渲染与 OCR                                                        │
│                                                                           │
│   pypdfium2 → 每页渲染为图片（200 DPI）                                   │
│       │                                                                   │
│       ▼                                                                   │
│   PaddleOCR（GPU）→ 识别文字 + 定位包围框（drop_score = 0.80）             │
│       │                                                                   │
│       ▼                                                                   │
│   页面分类                                                                 │
│   ┌───────────────────────────────────────────────────────────────────┐  │
│   │  Text  │  Table  │  TOC  │  Cover  │  Figure  │  Form            │  │
│   └───┬────────┬──────────────────────────────────────────────────────┘  │
│       │        │                    ↑                                    │
│       │        │        TOC / Cover / Figure → 跳过，不参与 chunking     │
└───────┼────────┼────────────────────────────────────────────────────────┘
        │        │
        │        │
        ▼        ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  F1  文本预处理                                                            │
│                                                                           │
│   多栏检测                                                                 │
│   同一 y-band 内水平间距 ≥ 60px → 拆分为独立列                             │
│   按 (y_min, x_min) 排序 → 防止左右列文字混排                              │
│                                                                           │
│   表格提取                                                                 │
│   ① 有边框表格 → OpenCV 格线检测 → 重建 Markdown pipe table               │
│   ② 无边框表格 → OCR 文本框 x/y 聚类 → 对齐重建                           │
│   ③ 复杂表格   → GPT-4o-mini（via Portkey）辅助结构化                     │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  F2  结构化分块（Chunking）                                                │
│                                                                           │
│   标题检测                                                                 │
│   ① 数字编号  1.2.3 / Chapter 2                                           │
│   ② 关键词    SECTION / PART / APPENDIX                                   │
│   ③ 位置推断  字号大 + 行短 + title_like 启发式                            │
│       │                                                                   │
│       ▼                                                                   │
│   GPT-4o-mini 标题规范化                                                  │
│   修复跨页章节边界 / 统一标题格式                                           │
│       │                                                                   │
│       ▼                                                                   │
│   分块                                                                    │
│   目标 800–1200 tokens，overlap 120 tokens                                │
│   < 250 tokens 的碎片 → 合并到邻近同章节 chunk                             │
│   表格 chunk 独立，不与正文合并                                             │
│       │                                                                   │
│       ▼                                                                   │
│   输出：final_chunks_vX.jsonl                                             │
│   共 10,119 chunks（206 文档）                                             │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  F3  LLM 分类（GPT-4o-mini via Portkey）                                  │
│                                                                           │
│   为每条 chunk 打标：                                                      │
│   primary_tag     → 8 类（action_policy / funding_program / ...）         │
│   policy_score    → 0.0–1.0（与政策内容的相关性）                          │
│   retrieval_tier  → primary / secondary / low_priority / exclude          │
└──────────────────┬────────────────────────────┬────────────────────────────┘
                   │                            │
                   ▼                            ▼
┌──────────────────────────────┐  ┌────────────────────────────────────────┐
│  F3  向量化 + Qdrant 入库     │  │  F3  BM25 索引                          │
│                              │  │                                        │
│  Gemini Embedding            │  │  rank_bm25（BM25Okapi）                 │
│  gemini-embedding-001        │  │  bm25_index.pkl    ← 打分算法           │
│  3072 维（via Portkey）       │  │  bm25_corpus_ids.pkl ← 位置→Qdrant ID  │
│       │                      │  │                                        │
│       ▼                      │  └────────────────────────────────────────┘
│  Qdrant Local DB             │
│  10,030 条（89 条 exclude）   │
│  Payload 字段：               │
│  policy_level / tribe_list   │
│  county_list / city_list     │
│  state_list / retrieval_tier │
└──────────────────────────────┘
```

---

## 2. Query Pipeline（在线，每次用户请求）

```
         用户在 Streamlit 界面操作
              │
    ┌─────────┴──────────────────────────────┐
    │                                        │
    ▼                                        ▼
┌─────────────────────────┐      ┌─────────────────────────────┐
│  地图点击（最高优先级）   │      │  下拉框选择（次高优先级）     │
│                         │      │                             │
│  folium 地图标记坐标     │      │  State / Tribe / County     │
│      │                  │      │  City 四个 selectbox        │
│      ▼                  │      └──────────────┬──────────────┘
│  shapely point-in-      │                     │
│  polygon 判断           │           ┌──────────┴──────────┐
│  → tribe / county /     │           │  冲突检测            │
│    city / state         │           │  地图 vs 下拉框      │
│  → 同步更新下拉框        │           │  若冲突 → 禁止搜索   │
└────────────┬────────────┘           └──────────┬──────────┘
             │                                   │
             └──────────────┬────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │  查询文本（最低优先级）        │
              │                             │
              │  规则匹配：词表直接查找        │
              │  tribe / county / city /    │
              │  state 名称                 │
              │      │                      │
              │      ▼                      │
              │  若无命中 → claude-haiku    │
              │  从 query 提取地理实体        │
              └─────────────┬───────────────┘
                            │
                            ▼ Active Geo
                            │ {tribes, counties, cities, states}
                            │
                            ▼
              ┌─────────────────────────────┐
              │  确定起始检索层级            │
              │                             │
              │  有 tribes  → tribe         │
              │  有 cities  → city          │
              │  有 counties→ county        │
              │  有 states  → state         │
              │  都没有     → federal only  │
              └─────────────┬───────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│  F4  分层级并行检索（search_all_levels）                                   │
│                                                                           │
│  ThreadPoolExecutor：各层同时发起，不等上一层完成                           │
│                                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Tribe 层    │  │ County 层   │  │ State 层    │  │ Federal 层  │    │
│  │             │  │             │  │             │  │             │    │
│  │ policy_level│  │ policy_level│  │ policy_level│  │ policy_level│    │
│  │ = Tribe     │  │ = County    │  │ = State     │  │ = Federal   │    │
│  │ + tribe_list│  │ +county_list│  │ +state_list │  │ （无地理    │    │
│  │   filter    │  │   filter    │  │   filter    │  │   限制）    │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │            │
│         ▼                ▼                ▼                ▼            │
│    ┌─────────────────────────────────────────────────────────────┐      │
│    │  每层独立执行：                                               │      │
│    │  Qdrant 向量检索（top_k 条）                                  │      │
│    │        +                                                     │      │
│    │  BM25 检索（top_k 条，ID 白名单限制）                          │      │
│    │        ↓                                                     │      │
│    │  RRF 融合（k=60）                                            │      │
│    │  + retrieval_tier 加权（×1.3 / ×1.0 / ×0.6）                │      │
│    │        ↓                                                     │      │
│    │  取前 top_n+2 条进入 rerank 候选池                            │      │
│    └─────────────────────────────────────────────────────────────┘      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ 所有层候选合并（最多 25 条）
                                ▼
              ┌─────────────────────────────┐
              │  claude-haiku-4-5 重排       │
              │                             │
              │  一次 API 调用              │
              │  对所有候选 0–10 打分        │
              │  → 归一化为 cross_score 0–1 │
              └─────────────┬───────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │  has_results 判断（每层）    │
              │                             │
              │  ✅ 有结果：                 │
              │  结果数 > 0                  │
              │  AND max(cross_score) ≥ 0.2 │
              │                             │
              │  ❌ 无结果：                 │
              │  结果为空 OR 分数全低        │
              └──────┬──────────────────────┘
                     │
          ┌──────────┴───────────────────────────────────┐
          │                                              │
          ▼                                              ▼
┌──────────────────────────┐              ┌─────────────────────────────┐
│  gen_input 构建           │              │  展示层（底部 chunks）        │
│                          │              │                             │
│  每个 ✅ 层取             │              │  🏘️ Tribal — Navajo Nation  │
│  前 N 条（slider 控制）   │              │  ✅ [1] score=0.82 ...      │
│  合并排序                 │              │  ✅ [2] score=0.71 ...      │
│                          │              │                             │
│  + level_summary         │              │  🗺️ County — Apache         │
│  {level → geo_label,     │              │  ❌ No relevant policies.   │
│   has_results}           │              │                             │
└──────────┬───────────────┘              │  📍 State — Arizona         │
           │                             │  ✅ [3] score=0.65 ...      │
           ▼                             │                             │
┌──────────────────────────┐             │  🏛️ Federal                 │
│  claude-sonnet-4-6 生成  │             │  ✅ [4] score=0.54 ...      │
│                          │             └─────────────────────────────┘
│  System prompt：         │
│  按层级结构化作答         │
│  + 强制 [Doc N] 溯源     │
│                          │
│  User message：          │
│  query + context chunks  │
│  + level_summary         │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  结构化答案                                           │
│                                                      │
│  **Tribal (Navajo Nation):**                         │
│  Water rights are governed by... [Doc 1][Doc 2]      │
│                                                      │
│  **County (Apache County):**                         │
│  No relevant county-level policies were found.       │
│                                                      │
│  **State (Arizona):**                                │
│  The Arizona Water Management Plan states... [Doc 3] │
│                                                      │
│  **Federal:**                                        │
│  EPA Region 6 Climate Adaptation... [Doc 4]          │
│                                                      │
│  Sources:                                            │
│  [Doc 1] Navajo Nation Water Rights Settlement ...   │
│  [Doc 3] Arizona Water Management Plan — https://... │
└──────────────────────────────────────────────────────┘
```

---

## 3. 地理层级与搜索方向

```
                    最宽泛
                       │
            ┌──────────▼──────────┐
            │   🏛️  Federal        │   policy_level = Federal
            │   EPA / FEMA / ...   │   无地理过滤，全库
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   📍  State          │   policy_level = State
            │   Arizona / NM / OK  │   state_list 过滤
            └──────────┬──────────┘
                       │
            ┌──────────▼──────────┐
            │   🗺️  County         │   policy_level = County
            │   Apache / Navajo... │   county_list 过滤
            └──────┬─────────┬────┘
                   │         │
        ┌──────────▼──┐   ┌──▼──────────────┐
        │ 🏙️  City    │   │ 🏘️  Tribe        │
        │ Flagstaff..│   │ Navajo Nation... │
        │ city_list  │   │ tribe_list 过滤   │
        └────────────┘   └─────────────────┘
                    最精确

    搜索方向：从用户指定层级 → 向上
    ───────────────────────────────
    用户问 Tribe  → 搜 Tribe + County + State + Federal
    用户问 County → 搜          County + State + Federal
    用户问 State  → 搜                   State + Federal
    用户问 Federal→ 搜                          Federal
```

---

## 4. 关键文件对应关系

```
用户操作
    │
    ├── Climate_Policy_RAG_Test.py   ← Streamlit UI，搜索流程，答案展示
    │       │
    │       ├── map_utils.py         ← folium 地图，point-in-polygon，冲突检测
    │       ├── retriever.py         ← search_all_levels()，geo 提取，RRF，rerank
    │       └── generator.py        ← build_context()，generate_answer()
    │
数据文件
    │
    ├── qdrant_storage/              ← 向量数据库（10,030 条）
    ├── bm25_index.pkl               ← BM25 打分算法
    ├── bm25_corpus_ids.pkl          ← BM25 位置 → Qdrant ID 映射
    ├── GeoJson_Data/                ← 10 个 GeoJSON（state/tribes/counties/cities）
    └── pdf_data/metadata/
            ├── policy_metadata_4.csv        ← 文档元数据（geo 字段）
            └── Policy Data Sheet - ....csv  ← URL → 可读政策名映射
```
