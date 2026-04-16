"""
migrate_qdrant_to_cloud.py
将本地 Qdrant 数据迁移到 Qdrant Cloud。

运行前请先设置环境变量：
    $env:QDRANT_URL     = "https://xxxx.aws.cloud.qdrant.io"
    $env:QDRANT_API_KEY = "your-qdrant-cloud-api-key"

然后运行：
    python migrate_qdrant_to_cloud.py
"""

import os
import sys
import time

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

QDRANT_LOCAL_PATH = os.path.join(_BASE_DIR, "qdrant_storage")
QDRANT_CLOUD_URL  = os.environ.get("QDRANT_URL", "")
QDRANT_CLOUD_KEY  = os.environ.get("QDRANT_API_KEY", "")
COLLECTION_NAME   = "climate_policy"
BATCH_SIZE        = 100   # 每批上传 100 条

if not QDRANT_CLOUD_URL or not QDRANT_CLOUD_KEY:
    print("ERROR: 请先设置 QDRANT_URL 和 QDRANT_API_KEY 环境变量。")
    sys.exit(1)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

print("连接本地 Qdrant...")
local = QdrantClient(path=QDRANT_LOCAL_PATH)

print(f"连接 Qdrant Cloud: {QDRANT_CLOUD_URL}")
cloud = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_CLOUD_KEY)

# 获取本地 collection 信息
local_info = local.get_collection(COLLECTION_NAME)
vector_size = local_info.config.params.vectors.size
print(f"本地向量维度: {vector_size}")
print(f"本地 chunk 总数: {local_info.points_count}")

# 在云端创建（或确认已存在）同名 collection
existing = [c.name for c in cloud.get_collections().collections]
if COLLECTION_NAME not in existing:
    print(f"在云端创建 collection '{COLLECTION_NAME}'...")
    cloud.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    print("Collection 创建成功。")
else:
    cloud_info = cloud.get_collection(COLLECTION_NAME)
    print(f"云端 collection 已存在，当前有 {cloud_info.points_count} 条。")

# 分批迁移
offset = None
total_uploaded = 0
print("\n开始迁移...")

while True:
    result, next_offset = local.scroll(
        collection_name=COLLECTION_NAME,
        limit=BATCH_SIZE,
        offset=offset,
        with_vectors=True,
        with_payload=True,
    )

    if not result:
        break

    points = [
        PointStruct(id=p.id, vector=p.vector, payload=p.payload)
        for p in result
    ]
    cloud.upsert(collection_name=COLLECTION_NAME, points=points)
    total_uploaded += len(points)
    print(f"  已上传 {total_uploaded} 条...", end="\r")

    if next_offset is None:
        break
    offset = next_offset
    time.sleep(0.1)  # 避免触发速率限制

print(f"\n迁移完成！共上传 {total_uploaded} 条向量。")

# 验证
cloud_info = cloud.get_collection(COLLECTION_NAME)
print(f"云端验证：{cloud_info.points_count} 条（本地：{local_info.points_count} 条）")

local.close()
cloud.close()
