# 产品需求文档（PRD）
## 美洲原住民气候政策智能问答系统（Climate Policy RAG）

**版本：** v0.9
**日期：** 2026-04-11
**状态：** 进行中

---

## 一、项目背景

美国亚利桑那州（AZ）、新墨西哥州（NM）、俄克拉荷马州（OK）三州境内居住着大量美洲原住民社区。这些社区正面临气候变化带来的切实威胁（干旱、洪灾、野火、热浪等），但由于信息获取渠道受限，许多居民无法高效地找到与自身处境相关的气候政策、减灾计划或资金申请渠道。

本项目旨在构建一个基于检索增强生成（RAG）技术的问答系统，将多层级、多来源的气候相关政策文件转化为可交互的知识库，帮助原住民居民快速定位适用于其部落、县、市或州的政策信息。

---

## 二、目标用户

**主要用户：** 居住在 AZ、OK、NM 三州的美洲原住民居民

**典型使用场景：**
- "我住在 Navajo 部落，我们这里水源污染严重，有没有相关的水资源政策或资金计划？"
- "我想改善 Apache 县的野火风险，我应该联系哪个政府部门申请资金支持？"
- "我们部落想推进太阳能项目，有哪些州或联邦层面的支持政策？"

**用户特征：**
- 语言：英语（当前版本，暂不支持其他语言）
- 技术背景：不做假设，界面需友好易用
- 需求特点：高度地理敏感（部落、县、市层级精准匹配）、信任要求高（需有明确来源）

---

## 三、数据概况

### 3.1 本地 PDF 文件（主数据源）

**数量：** 129 份（OCR 自动 chunking）+ 若干需人工 chunking（见 3.2）
**元数据文件：**
- `pdf_data/metadata/policy_metadata_4.csv`（129 份，OCR chunker 使用）
- `pdf_data/metadata/policy_metadata_4_HPC.csv`（同上，路径前缀改为 `/scratch/xt2284/`，供 NYU HPC 使用）
- `pdf_data/metadata/unstructured_metadata.csv`（需人工 chunking 的文件，`manual_chunker.py` 使用）

**本地路径：** `pdf_data/data/{State}/{Level}/{filename}.pdf`

**目录结构：**
```
pdf_data/data/
├── Arizona/
│   ├── State/
│   ├── County/
│   ├── City/
│   └── Tribe/
├── New Mexico/
│   ├── State/
│   ├── County/
│   ├── City/
│   └── Tribe/
└── Oklahoma/
    ├── State/
    ├── County/
    ├── City/
    └── Tribe/
```

**元数据字段：** `policy_id`（SHA1）、`source_url`、`file_path`、`state_list`、`county_list`、`city_list`、`tribe_list`、`policy_level`（State/County/City/Tribe）、`policy_type`

**主要文件类型分布：**

| 政策类型 | 分布层级 | 说明 |
|---|---|---|
| Hazard Mitigation Plan | County 为主 | 最多，几乎每个县一份 |
| Climate Action Plan | State/County/City/Tribe | 气候行动计划 |
| Water Plan / Drought Plan | State/County/City | 水资源相关 |
| Comprehensive Plan | County/City | 综合规划（含土地利用、环境章节）|
| Emergency Plan | State/County | 应急预案 |
| Energy Plan | State/Tribe | 能源政策（新能源为主）|
| Wildfire Plan | County | 野火防治 |
| Environmental Code | Tribe | 部落环境法规 |
| Climate Adaptation Plan | Tribe | 部落气候适应计划 |

### 3.2 需人工 Chunking 的文件

排版复杂、OCR 效果差的 PDF 单独列入 `unstructured_metadata.csv`，使用 `manual_chunker.py`（Streamlit 工具）进行人工分段。该工具输出与 `OCR_chunker_full_v11_6.py` 完全相同的 JSONL 格式。

**运行方式：**
```bash
& E:/tool/anaconda/envs/pdfocr-gpu/python.exe -m streamlit run manual_chunker.py
```

### 3.3 文档挑战（已知问题）

1. **多栏排版（高优先级）：** 部分 PDF 采用双栏或多栏布局。当前 OCR 管道的行合并逻辑（按 y 坐标聚类）会将同一水平位置的两列内容错误合并，导致文本语序混乱，是当前 chunking 失败的主要原因之一。
2. **复杂表格：** 核心信息以表格呈现（灾害风险矩阵、资金分配表等），包含无边框表格、合并单元格。当前基于 OpenCV 的格线检测对有边框表格有效，对无边框表格效果较差。
3. **格式高度异构：** 196 份文件来自不同政府机构，无统一写作规范，标题层级、章节编号格式差异大。
4. **扫描版/图片 PDF：** 部分文件为扫描版，PaddleOCR 识别率在复杂排版页面表现一般。
5. **图片为主的页面：** 部分页面主要内容为图表、地图，OCR 无实质文本，当前标记为 `is_figure=True` 并跳过。

---

## 四、系统功能需求

### 4.1 核心功能

#### F1：文档解析与预处理（**当前瓶颈**）

- **F1.1** PDF 渲染：使用 `pypdfium2` 将每页渲染为图像（200 DPI），作为 OCR 输入
- **F1.2** OCR：使用 PaddleOCR（GPU 模式，`drop_score=0.80`）进行文本识别
- **F1.3** **多栏检测（已实现）：** `merge_ocr_items_into_lines_with_pos` — 同 y-band 内水平间距 ≥ 60px 的 OCR 项拆分为独立列，按 (y_min, x_min) 排序，防止左右列文字混排
- **F1.8** **人工 Chunking 工具（已实现）：** `manual_chunker.py` — Streamlit 界面，读取 `unstructured_metadata.csv`，支持手动分段并输出统一 JSONL 格式
- **F1.4** 表格提取：
  - 有边框表格：OpenCV 格线检测（`detect_table_cells_from_image`）→ 按行列重建 Markdown 表格
  - 无边框表格：基于 OCR 文本框 x/y 聚类（`detect_table_block_from_ocr`）→ 重建对齐文本
  - 复杂/不规则表格：使用 LLM（GPT-4o-mini via Portkey）辅助结构化（调用频次受限）
- **F1.5** 页面分类（已实现）：TOC / Cover / Figure / Form / Table / Text，特殊页不参与语义 chunking
- **F1.6** OCR 质量跟踪：记录每页置信度，低质量页（平均置信度 < 0.75）标记 `low_ocr_quality=True`
- **F1.7** 每个文本块保留完整 metadata（见第五节 Chunk Schema）

#### F2：结构化分块（Chunking）

- **F2.1** 标题检测（已实现）：支持数字编号（`1.2.3`）、关键词（`CHAPTER/SECTION/PART`）、无编号标题（位置+格式推断）
- **F2.2** LLM 辅助标题修正（已实现）：GPT-4o-mini 规范化标题、修复跨页章节边界
- **F2.3** 目标 chunk 大小：800-1200 tokens，overlap 120 tokens
- **F2.4** 小 chunk 合并（已实现）：低于 250 tokens 的 chunk 向邻近同章节 chunk 合并
- **F2.5** 跨页 chunk：同一章节内容跨页时合并为一个 chunk，保留所有涉及页码
- **F2.6** 表格 chunk 独立处理：表格作为独立 chunk，不与正文文本合并

#### F3：向量化与索引

- **F3.1** 嵌入模型：`gemini-embedding-001`（via NYU Portkey @vertexai，**3072 维**）✅ 已完成
- **F3.2** 向量数据库：**本地 Qdrant**（`qdrant_storage/`）✅ 已完成，10,030 条向量
- **F3.3** 实际存储：10,119 chunks（206 文档）→ 去除 89 条 exclude → **10,030 条入库**
- **F3.4** Payload 索引字段（已建立）：`primary_tag`、`retrieval_tier`、`policy_score`、`policy_level`、`policy_type`、`is_table`、`is_appendix`
- **F3.5** LLM 分类（已完成）：每条 chunk 附加 `primary_tag`、`secondary_tags`、`policy_score`、`retrieval_tier` 四个字段（8 类标签体系）
- **F3.6** 关键词索引：BM25（`rank_bm25`，持久化为 `bm25_index.pkl` + `bm25_corpus_ids.pkl`）— ✅ 已完成

#### F4：查询理解与检索

- **F4.1** 地理实体识别：从用户查询中提取部落名、县名、州名；部落名需经过别名规范化（见第六节）
- **F4.2** 主题提取：识别查询涉及的主题（水/野火/资金/能源/气候适应等）
- **F4.3** 混合检索：向量检索 + BM25，RRF 融合（k=60）+ retrieval_tier 加权（primary×1.3 / secondary×1.0 / low_priority×0.6）+ claude-haiku 批量打分重排（0-10，归一化为 cross_score 0-1）
- **F4.4** 分层级检索策略（`search_all_levels()`）：从用户指定的最具体地理层级开始，**向上**搜索所有层级，不向下：
  - 用户指定 Tribe → 搜索：Tribe + County + State + Federal
  - 用户指定 City  → 搜索：City + County + State + Federal
  - 用户指定 County → 搜索：County + State + Federal
  - 用户指定 State  → 搜索：State + Federal
  - 无 geo 识别   → 搜索：Federal only
  - 每层独立过滤（`policy_level` 字段 + 地理字段双重过滤），并行执行（ThreadPoolExecutor）
  - 每层 `has_results = 结果数量 > 0 AND max(cross_score) ≥ 0.2`
- **F4.5** 每条 chunk 附 `geo_level` 标签（tribe / city / county / state / federal），用于展示层分组和生成层引导
- **F4.6** 地理来源优先级：地图标记 > 下拉框选择 > 查询文本自动提取；地图与下拉框冲突时禁止检索并提示用户
- **F4.7** 相关性门控：所有层级最高 `cross_score` 均 < 0.2 时不触发答案生成

#### F5：答案生成

- **F5.1** LLM：`claude-sonnet-4-6`（via NYU Portkey @vertexai）
- **F5.2** **溯源强制要求：** 每一条关键陈述附 `[Doc N]` 引用，答案末尾附 Sources 列表（文件名 + URL）
- **F5.3** 答案风格：按地理层级分段展示，每段格式为 `**[Level] ([Location]):** ...`；若某层无相关政策则注明 "No relevant [level]-level policies were found."；`level_summary` 字典作为额外上下文传入生成模型
- **F5.4** 不命中处理：`cross_score` 门控（< 0.2）+ LLM system prompt 要求无信息时明确声明
- **F5.5** 文档名显示：通过 URL 匹配 Policy Data Sheet CSV，将内部哈希文件名替换为可读政策名称

### 4.2 硬性约束

| 约束 | 实现方式 |
|---|---|
| 溯源可追踪 | 每个 chunk 携带 `policy_id`、`source_url`、`page` 字段，LLM prompt 强制要求引用 |
| 不捏造信息 | Prompt 明确："Only use information from the provided context. If the context doesn't contain relevant information, say so." |
| 地理精度 | 检索前元数据过滤；答案中标注文件适用地理范围 |
| 本地部署 | 向量数据库（Qdrant local）、BM25 索引均存储在本地 E 盘 |

---

## 五、Chunk 数据格式（Schema）

```json
{
  "chunk_id": "f1c94ead_p012_c03",
  "policy_id": "f1c94ead9b4cd569ba519154a50286736d9737f6",
  "source_url": "https://...",
  "doc_title": "2004 Arizona Drought Preparedness Plan",
  "file_path": "E:/2026_capstone/policy_data/pdf_data/data/Arizona/State/...",
  "policy_level": "State",
  "policy_type": "Drought Plan",
  "state_list": ["Arizona"],
  "county_list": [],
  "city_list": [],
  "tribe_list": [],
  "section": "2.1: Drought Indicators",
  "section_path": ["2: Assessment Methods", "2.1: Drought Indicators"],
  "pages": [12, 13],
  "tokens": 847,
  "text": "...",
  "is_table": false,
  "is_appendix": false,
  "is_toc": false,
  "is_figure": false,
  "low_ocr_quality": false,
  "page_type": "text"
}
```

---

## 六、部落名称规范化

不同文件中同一部落可能有多种写法。在元数据过滤和地理匹配时需统一规范。以下为已知需处理的别名：

| 规范名称 | 已知别名 |
|---|---|
| Navajo Nation | Navajo, Diné, Navajo Tribe |
| White Mountain Apache Tribe | White Mountain Apache, Fort Apache |
| Hopi Tribe | Hopi |
| Havasupai Tribe | Havasupai |
| San Carlos Apache Tribe | San Carlos Apache |
| Tohono O'odham Nation | Tohono O'odham, Papago |
| Jicarilla Apache Nation | Jicarilla Apache |
| Mescalero Apache Tribe | Mescalero Apache |
| Pueblo of Santa Ana | Santa Ana Pueblo |
| Pueblo of Sandia | Sandia Pueblo |
| Tesuque Pueblo | Pueblo of Tesuque |

*此列表应在 chunking/indexing 阶段持续扩充。规范化规则存储为 JSON 文件，检索时用于别名展开查询。*

---

## 七、技术架构

### 7.1 总体流程

> 详细 Mermaid 流程图见 `WORKFLOW.md`

```
━━━━━━━━━━━━━━━━━ INGESTION（离线，一次性）━━━━━━━━━━━━━━━━━

[196 个 PDF] ──pypdfium2──▶ [PaddleOCR GPU] ──▶ [页面分类]
                                                      │
                          ┌───────────────────────────┤
                     Text/Table pages           Figure/TOC/Cover → skip
                          │
              ┌──多栏检测──┴──表格提取 (OpenCV + LLM)──┐
              ▼                                         ▼
     [标题检测 + GPT-4o-mini 规范化]          [Table chunks]
              │
              ▼
     [Chunking 800-1200 tokens]
              │
              ▼
     [GPT-4o-mini 分类] → primary_tag / policy_score / retrieval_tier
              │
     ┌────────┴────────┐
     ▼                 ▼
[Gemini Embedding]  [BM25 Index]
[Qdrant 10,030 条]  [bm25_index.pkl]

━━━━━━━━━━━━━━━━━ QUERY（在线，每次请求）━━━━━━━━━━━━━━━━━

[地图点击] ──┐
[下拉框]   ──┼──▶ [Active Geo] ──▶ [起始层级确定]
[查询文本]  ──┘         │
                    冲突检测
                        │
                ┌───────▼──────────────────────────────┐
                │  search_all_levels() — 并行           │
                │  Tribe │ County │ State │ Federal      │
                │  各层：Qdrant + BM25 + RRF            │
                └───────────────┬──────────────────────┘
                                │
                    [claude-haiku rerank 0-10]
                                │
                    [has_results 判断，分层]
                                │
                ┌───────────────┴────────────────┐
                ▼                                ▼
        [gen_input]                      [展示层分组]
        N chunks/level                   ✅/❌ per level
                │
        [claude-sonnet-4-6]
        + level_summary prompt
                │
        [分层级结构化答案]
        **Tribal (X):** ...
        **State (Y):** ...
        **Federal:** No policies found.
        + Sources
```

### 7.2 关键技术选型

| 模块 | 选型 | 备注 |
|---|---|---|
| PDF 渲染 | `pypdfium2` | 已集成，速度快 |
| OCR | `PaddleOCR` | 已集成，GPU 加速 |
| 表格提取（有边框）| `OpenCV` | 已集成 |
| 表格提取（复杂）| `GPT-4o-mini via Portkey` | LLM fallback，限频 |
| 嵌入模型 | `gemini-embedding-001`（NYU Portkey @vertexai，dim=3072）| ✅ 已确认，10,030 条已入库 |
| 向量数据库 | `Qdrant`（本地模式）| 支持 payload 过滤，~2-4 GB |
| 关键词检索 | `rank_bm25` | 本地 |
| 重排器 | `claude-haiku-4-5`（via Portkey，批量 0-10 打分） | 无本地模型依赖，适合无 GPU 展示环境 |
| LLM（答案生成）| `claude-sonnet-4-6` via Portkey @vertexai | 强制 [Doc N] 溯源 |
| LLM（地理提取/重排）| `claude-haiku-4-5` via Portkey @vertexai | 速度快，成本低 |
| 地图组件 | `folium` + `streamlit-folium` | 交互式 GeoJSON 地图，点击放标记 |
| 空间计算 | `shapely`（point-in-polygon）| 本地计算，无 API |

---

## 八、当前进度与问题

### 8.1 已完成

- [x] 数据收集：196 份本地 PDF + 元数据 CSV
- [x] OCR 管道框架（v11.6）：pypdfium2 渲染 + PaddleOCR + 页面分类 + 标题检测 + 表格检测 + LLM 辅助
- [x] Chunk schema 设计（含完整地理 + 政策元数据）
- [x] Portkey/GPT-4o-mini 集成
- [x] `download_unstructured.py` 编写完成（待执行）

### 8.2 Chunking 问题深度分析（2026-03-09）

通过阅读 `final_chunks_v11_6.jsonl` 和 `page_chunks_debug_v11_6.jsonl`，确认以下具体问题：

| # | 问题 | 根因 | 影响 |
|---|---|---|---|
| B1 | **只处理 1 个 PDF** | `__main__` 设置 `sample_from_metadata_n=1`，是测试参数 | 所有历史运行结果只含 AZ 干旱计划 |
| B2 | **批处理无 try/except** | 一个 PDF 报错会中断整批 | 实际运行中即使改为全量也可能中途终止 |
| B3 | **贡献者名字成为章节名** | 标题检测对致谢页过于激进（`title_like` 函数匹配人名/委员会名）| 5~65 token 的无意义 chunk 大量生成 |
| B4 | **section_path 在 final chunks 全为 null** | `split_text_into_chunks` 未接收并设置 `section_path` 参数 | 元数据不完整，影响检索过滤 |
| B5 | **空表格输出 "Col1...Col37"** | 表格边框检测到但 OCR 文本为空，自动生成列名 | 无内容的 chunk 占用空间、污染语义 |

### 8.3 Chunking 修复记录（已全部应用于 v11_6）

| # | 修复 | 具体改动 |
|---|---|---|
| **B1+B2** | 批处理 & 错误处理 | `__main__` 扫描全量 PDF；`run()` 循环加 try/except |
| **B3** | 碎片 section 合并 | 新增 `consolidate_tiny_blocks()`：< 200 token block 合并入上一个 block |
| **B4** | section_path 传递 | `split_text_into_chunks` 新增 `section_path` 参数并传入 `b["section_key"]` |
| **B5** | 空表格退化守卫 | 全列名为 Col\d+ 且无数据行时退回原始 OCR 文本 |
| **B6** | 多栏排版 | `merge_ocr_items_into_lines_with_pos`：同 y-band 内水平间距 ≥ 60px 的 OCR 项拆分为独立列，按 (y_min, x_min) 排序 |
| **B7** | Markdown 表格格式 | 所有表格改为 `\| col \| col \|` pipe table 格式，新增 `_flatten_cell` / `_build_markdown_table` |
| **B8** | final chunk 布尔字段全 null | `split_text_into_chunks` 根据 `[TABLE]` 标记推断并填充 `is_table`、`page_type` 等 5 个字段 |
| **B9** | 附录 section 名重复 | `build_scientific_chunks` 中附录 chunk section 改为 `"APPENDIX X > 子标题"` 格式 |
| **B10** | 长标题被 title_like 拒绝 | 词数限制 8 → 12，允许 2-3 个小写介词（cap_ratio ≥ 70%） |
| **B11** | Section 级 running header 未 blacklist | `HEADER_FREQ_FRAC` 0.67 → 0.40（覆盖仅出现在部分章节的页眉）|
| **B12** | Org chart 被误识别为 table | 新增稀疏度检测：sparsity > 65% + 无数字 → 回退为 prose |
| **B13** | title_like 误检正文句子 | 新增：末尾 stop word 过滤、句子开头词过滤（>6词）、尾数字过滤、冒号标签标题支持 |
| **B14** | HEADING_WORDY 接受正文为标题 | CHAPTER/SECTION/PART 模式匹配后追加 `title_like(title)` 校验（词数>2时）|
| **B15** | HEADING_NUMERIC 多级编号接受正文 | `1.2.3` 形式的多级编号标题追加 `title_like(title)` 校验 |
| **B16** | 垃圾表格 chunk（页眉/logo）| `merge_tiny_final_chunks` 预过滤：`is_table=True` + tokens≤30 + 内容匹配 `_PAGE_HEADER_RE` → 删除 |
| **B17** | 全大写冒号标题未检测 | `title_like` 新增：before 部分≤5词且全大写时，跳过冒号拒绝逻辑，按完整行校验 |
| **B18** | 宽稀疏图表页被提取为乱码表格 | 无边框表格守卫：≥6列 + 空格率>55% + 平均内容<20字符 → 退回 prose |
| **B19** | TOC 中的 Annex 条目触发附录检测 | `detect_appendix_on_page` 新增守卫：同页≥2行以 `Annex/Appendix X:` 开头 → 判定为目录页，返回 False |

### 8.4 已知未解决问题

| # | 问题 | 说明 | 优先级 |
|---|---|---|---|
| **B11-残留** | Section 级 running header（低频）| "Governor's Drought Task Force" 仅出现在 19% 页面（低于 40% 阈值），仍被识别为 section name。需实现连续页检测（5+ 连续页同一文本 → blacklist） | 低 |
| **B20** | 机构名误识别为 section | "Arizona Department of Water Resources" 等机构名与标题在大小写上无法区分 | 低 |

### 8.5 第二轮分析结论（3 个 PDF，313 chunks）

| 指标 | 数值 | 说明 |
|---|---|---|
| 总 chunks | 313 | — |
| `is_table` null | 0 | B8 修复后全部填充 ✓ |
| section_path null | 0 | B4 修复 ✓ |
| `(no heading)` | 42 (13.4%) | 主要为整页表格和延续页，正常 |
| 含 `[TABLE]` chunks | 152 (49%) | hazard plan 表格密度正常 |
| token 中位数 | ~495 | 合理范围 |
| 附录子章节格式 | 37 chunks | B9 修复，格式为 "APPENDIX X > 子标题" ✓ |
| Criterion / Goal 类标题 | 可检测 | B13 修复后支持含冒号标签标题 ✓ |

---

## 八点五、测试界面功能（v0.9 更新）

`Climate_Policy_RAG_Test.py`（Streamlit）— 无侧边栏、无标签页，两栏布局：

### 布局结构

| 区域 | 内容 |
|---|---|
| 顶部标题栏 | 标题 + Portkey API Key 输入（内联，右侧）|
| 左栏（5份宽）| folium 地图（480px）+ 地理过滤下拉框（2×2）+ Reset + Advanced Settings |
| 右栏（6份宽）| 地理状态指示 + 查询输入 + Search/Clear 按钮 + 答案展示 |
| 底部全宽 | Retrieved Chunks，按地理层级分组展示 |

### 地理过滤（三来源，有优先级）

| 来源 | 方式 | 优先级 |
|---|---|---|
| 地图标记 | 左栏 folium 地图点击，point-in-polygon 识别 tribe/county/city/state | 最高 |
| 下拉框 | 左栏 State / County / Tribe / City 四个 selectbox（2×2 网格）| 中 |
| 查询文本 | 规则匹配 + claude-haiku 兜底，自动从 query 提取 | 最低 |

地图标记放置后四个下拉框自动同步；地图与下拉框冲突时禁止搜索并提示。

### 交互地图

- 基于 `folium` 渲染，`streamlit-folium` 集成（`returned_objects=["last_clicked"]` + `@st.fragment`）
- 图层：州边界 / 县边界 / AZ·NM·OK 部落边界（分组）/ 城市边界（默认隐藏）
- GeoJSON 来源：`GeoJson_Data/`（state/tribes/counties/cities，共 10 个文件）
- `map_utils.py`：`identify_region(lat, lon)`、`detect_conflict()`、`build_folium_map()`

### 检索结果展示（v0.9 新增分层）

- 按地理层级分组：Tribal / City / County / State / Federal
- 每层标注 ✅（has_results）或 ❌（无相关政策）
- 文档名：URL 匹配 Policy Data Sheet CSV → 可读政策名，fallback 清理哈希前缀
- 本地路径：HPC 路径（`/scratch/xt2284/`）自动转换为 `E:/2026_capstone/`
- 相关性分数：haiku 0-10 归一化为 0-1，所有层级最高分 < 0.2 时不生成答案

### Advanced Settings 滑块

| 滑块 | 默认值 | 含义 |
|---|---|---|
| Candidates per level (top_k) | 8 | 每层从 Qdrant+BM25 取多少候选 |
| Results per level (top_n) | 5 | 每层 rerank 后展示多少条 |
| Chunks per level sent to LLM | 2 | 每层送给 sonnet 多少条（上限 = top_n）|

---

## 九、已解决决策项 & 未解决决策项

### 已解决

| 编号 | 问题 | 决策 |
|---|---|---|
| D1 | 嵌入模型选择 | **`gemini-embedding-001`（dim=3072，via NYU Portkey @vertexai）**，已全量入库 10,030 条 |
| D2 | 联邦数据是否纳入 RAG | **已纳入**（v0.9）。Federal 作为分层检索的最高层，通过 `policy_level=Federal` 过滤，独立展示，不与低层级混排，避免污染 |
| D3 | 系统入口形态 | **Streamlit Web 界面**（`Climate_Policy_RAG_Test.py`），测试阶段保留调试信息 |
| D5 | 重排器选型 | **claude-haiku LLM 打分**（替代本地 cross-encoder），原因：最终展示环境可能无 GPU/本地模型 |
| D6 | 地理冲突处理 | **冲突时禁止检索 + 提示用户**，不自动猜优先级 |

### 待解决

| 编号 | 问题 | 当前倾向 | 待确认 |
|---|---|---|---|
| D4 | PaddleOCR 参数调优 | 当前 DPI=200，drop_score=0.80 | 是否对特定问题文件单独调高 DPI？ |
| D7 | ArcGIS Storymap 集成 | 队友提供 URL 后用 `st.components.v1.iframe` 嵌入 | 等待队友发布链接 |

---

## 十、成功指标

| 指标 | 目标 |
|---|---|
| 全量 PDF 处理完成率 | ≥ 95%（允许少量极度异常文件手动处理）|
| 答案溯源率 | 100% |
| 地理匹配准确率 | ≥ 90% |
| 表格内容覆盖率 | ≥ 80%（有结构的表格可被检索）|
| 空答率 | ≤ 10% |

---

## 十一、里程碑

| 阶段 | 目标 | 关键产出 | 状态 |
|---|---|---|---|
| **P0：Chunking 修复** | 全量 129 份 PDF（policy_metadata_4.csv）成功 chunk | `final_chunks_vX.jsonl`，全量诊断报告 | **19 个 bug 已修复（B1-B19），`sample_from_metadata_n=0` 运行全量** |
| **P0.5：人工 Chunking** | 排版复杂文件人工分段 | `unstructured_metadata.csv` chunks 合并入主 JSONL | `manual_chunker.py` 已就绪，**待执行** |
| **P1：向量化入库** | 全量 chunks 写入 Qdrant local + BM25 | 可查询的本地知识库 | ✅ **完成**（Qdrant 10,030 条 + BM25 索引）|
| **P2：检索系统** | 混合检索 + 地理过滤 + 重排 | `retriever.py` + `build_bm25.py` | ✅ **完成**（BM25+Qdrant+RRF+LLM rerank+geo_level 双层）|
| **P3：生成系统** | 接入 LLM，带溯源答案生成 | `generator.py` + `app.py` | ✅ **完成**（claude-sonnet-4-6，本地/区域分层答案）|
| **P3.5：界面迭代（甲方需求 R1–R5）** | 文档名可读化、路径修复、地图集成、地理扩展 | `Climate_Policy_RAG_Test.py` + `map_utils.py` | ✅ **完成**（2026-04-05）|
| **P3.6：分层级检索（R6）** | 从用户层级向上搜索所有层级，分层展示结果，Federal 纳入检索 | `retriever.search_all_levels()` + UI + generator | ✅ **完成**（2026-04-11）|
| **P4：评估迭代** | 构建评估集，计算 Recall@5 / MRR / NDCG | 评估报告 | 待开始 |
| **P5：界面封装** | 正式 Web 界面（去除测试调试信息）| 可交付原型 | 待开始（ArcGIS Storymap 待集成）|
