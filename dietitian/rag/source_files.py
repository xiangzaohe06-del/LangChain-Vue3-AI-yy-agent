"""Read a complete local source revision; fingerprint content, not timestamps."""
import hashlib
import json
from pathlib import Path


def source_fingerprint(files, sources):
    payload = {"files": files, "sources": {name: sources.get(name, {}) for name in files}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def read_directory(directory: Path, sources: Path):
    directory = directory.resolve(strict=True)
    manifest = json.loads(sources.read_text(encoding="utf-8-sig")) if sources.exists() else {}
    if not isinstance(manifest, dict):
        raise ValueError("Source manifest must be an object")
    documents, hashes, metadata = [], {}, {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError("Documents must stay inside the watched directory")
        name = path.relative_to(directory).as_posix()
        data = path.read_bytes()
        meta = manifest.get(name, {})
        if not isinstance(meta, dict):
            raise ValueError("Document metadata must be an object")
        documents.append((name, data, meta))
        hashes[name] = hashlib.sha256(data).hexdigest()
        metadata[name] = meta
    if not documents:
        raise ValueError("Watched directory is empty; previous knowledge is retained")
    return documents, source_fingerprint(hashes, metadata)
