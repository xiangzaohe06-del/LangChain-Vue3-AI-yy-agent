import argparse
import json
from pathlib import Path

from rag.store import knowledge_store, KnowledgeStore


def main():
    parser = argparse.ArgumentParser(description="发布、构建、检索营养知识库")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--lexical-only", action="store_true", help="仅构建 BM25，不调用 Embedding API")
    search = commands.add_parser("search")
    search.add_argument("query")
    commands.add_parser("status")
    commands.add_parser("sync-milvus", help="将当前快照已有向量导入Milvus并构建HNSW，不调用Embedding")
    publish = commands.add_parser("publish", help="上传完整资料目录到 COS，构建后切换当前版本")
    publish.add_argument("directory", type=Path)
    publish.add_argument("--sources", type=Path, help="可选的来源元数据 JSON 文件")
    publish.add_argument("--lexical-only", action="store_true")
    publish.add_argument("--stage", action="store_true", help="仅构建候选版本，通过验证后再激活")
    rollback = commands.add_parser("rollback", help="切换到指定 COS 索引版本")
    rollback.add_argument("version")
    commands.add_parser("versions", help="列出 COS 中保留的索引版本（包含未激活版本）")
    args = parser.parse_args()
    if args.command in {"publish", "rollback", "versions"}:
        from rag.cos_storage import CosKnowledgeStorage
        storage = CosKnowledgeStorage()
        store = KnowledgeStore(storage=storage)
        if args.command == "publish":
            result = store.publish_directory(args.directory, args.sources,
                vectors=not args.lexical_only, activate=not args.stage)
        elif args.command == "rollback":
            result = store.activate_version(args.version)
        else:
            result = storage.versions()
    elif args.command == "build":
        result = knowledge_store.rebuild(vectors=not args.lexical_only)
    elif args.command == "search":
        result = knowledge_store.search(args.query)
    elif args.command == "sync-milvus":
        result = knowledge_store.sync_milvus()
    else:
        result = knowledge_store.status()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
