"""COS-backed immutable datasets and index snapshots; runtime caches only in RAM."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import gzip
import json
import mimetypes
import os
import re
from pathlib import Path, PurePosixPath
from threading import RLock
import uuid

from qcloud_cos import CosConfig, CosS3Client
from qcloud_cos.cos_exception import CosServiceError


def encode_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_json(data: bytes):
    return json.loads(gzip.decompress(data) if data.startswith(b'\x1f\x8b') else data)


def checked_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or any(p in {"", ".", ".."} for p in name.split("/")):
        raise ValueError("Invalid knowledge document path")
    if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
        raise ValueError("Only Markdown, TXT and PDF documents are supported")
    return name


def checked_version(version: str) -> str:
    if len(version) != 32 or any(c not in "0123456789abcdef" for c in version):
        raise ValueError("Invalid knowledge version")
    return version


class CosKnowledgeStorage:
    def __init__(self, *, client=None, bucket=None, prefix=None):
        self._client = client
        self.bucket = bucket
        self.prefix = (prefix or os.getenv("RAG_COS_PREFIX", "knowledge/nutrition")).strip("/")
        if not self.prefix or "\\" in self.prefix or any(p in {"", ".", ".."} for p in self.prefix.split("/")):
            raise ValueError("RAG_COS_PREFIX must be a non-empty COS folder")
        self._cache = None
        self._lock = RLock()

    @property
    def client(self):
        # Lazy initialization keeps the app importable before credentials are configured.
        if self._client is None:
            secret_id, secret_key = os.getenv("COS_SECRET_ID"), os.getenv("COS_SECRET_KEY")
            region = os.getenv("COS_REGION")
            bucket = os.getenv("RAG_COS_BUCKET") or os.getenv("COS_BUCKET")
            app_id = os.getenv("COS_APP_ID")
            if not all((secret_id, secret_key, region, bucket)):
                raise RuntimeError("Configure COS_SECRET_ID, COS_SECRET_KEY, COS_REGION and COS_BUCKET locally")
            if not re.fullmatch(r"[a-z]{2}(?:-[a-z0-9]+)+", region):
                raise ValueError("COS_REGION must be a region code such as ap-nanjing, not an endpoint URL")
            if app_id and not bucket.endswith("-" + app_id):
                bucket += "-" + app_id
            if not bucket.rsplit("-", 1)[-1].isdigit():
                raise RuntimeError("COS_BUCKET must include the APPID suffix, or configure COS_APP_ID")
            self.bucket = bucket
            self._client = CosS3Client(CosConfig(
                Region=region, SecretId=secret_id, SecretKey=secret_key,
                Token=os.getenv("COS_SESSION_TOKEN"), Scheme="https",
                Timeout=int(os.getenv("RAG_COS_TIMEOUT", "120")),
            ))
        return self._client

    def _get(self, key: str, *, missing_ok=False) -> bytes | None:
        client = self.client
        try:
            response = client.get_object(Bucket=self.bucket, Key=key)
        except CosServiceError as error:
            if missing_ok and error.get_error_code() == "NoSuchKey":
                return None
            raise
        with closing(response["Body"].get_raw_stream()) as stream:
            if hasattr(stream, 'decode_content'):
                stream.decode_content = False
            return stream.read()

    def _put(self, key: str, data: bytes, content_type="application/json; charset=utf-8", *, compressed=False):
        client = self.client
        headers = {"ContentEncoding": "gzip"} if compressed else {}
        client.put_object(Bucket=self.bucket, Key=key, Body=data, EnableMD5=True,
                          ACL="private", ContentType=content_type, CacheControl="no-cache", **headers)

    def _dataset_key(self, dataset_id: str) -> str:
        return f"{self.prefix}/datasets/{checked_version(dataset_id)}/manifest.json"

    def _index_key(self, version: str) -> str:
        return f"{self.prefix}/indexes/{checked_version(version)}.json"

    def stage_directory(self, directory: Path, sources: Path | None = None) -> dict:
        """Upload and verify a new source snapshot without changing the active version."""
        directory = directory.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("Knowledge source must be a directory")
        manifest = json.loads(sources.read_text(encoding="utf-8")) if sources else {}
        def documents():
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".pdf"}:
                    continue
                if path.is_symlink() or not path.resolve().is_relative_to(directory):
                    raise ValueError("Documents must stay inside the import directory")
                filename = checked_name(path.relative_to(directory).as_posix())
                yield filename, path.read_bytes(), manifest.get(filename, {})
        return self.stage_documents(documents())

    def stage_documents(self, documents):
        """Freeze a complete source revision before parsing or embedding it."""
        dataset_id = uuid.uuid4().hex
        files, sources = {}, {}
        for filename, data, meta in documents:
            checked_name(filename)
            if filename in files:
                raise ValueError("Duplicate source document")
            key = f"{self.prefix}/datasets/{dataset_id}/documents/{filename}"
            self._put(key, data, mimetypes.guess_type(filename)[0] or "application/octet-stream")
            if digest(self._get(key)) != digest(data):
                raise ValueError("COS document checksum mismatch")
            files[filename] = {"key": key, "sha256": digest(data), "size": len(data)}
            sources[filename] = meta
        if not files:
            raise ValueError("No supported documents found; current knowledge remains unchanged")
        dataset = {"schema_version": 1, "dataset_id": dataset_id, "files": files,
                   "sources": sources}
        payload = encode_json(dataset)
        key = self._dataset_key(dataset_id)
        self._put(key, payload)
        if self._get(key) != payload:
            raise ValueError("COS dataset manifest verification failed")
        return {"dataset_id": dataset_id, "documents": len(files)}

    def source_manifest(self, dataset_id: str):
        dataset = json.loads(self._get(self._dataset_key(dataset_id)))
        if dataset.get("schema_version") != 1 or dataset.get("dataset_id") != dataset_id:
            raise ValueError("Invalid COS dataset manifest")
        return dataset

    def read_documents(self, dataset_id: str):
        dataset = self.source_manifest(dataset_id)
        for name, info in sorted(dataset["files"].items()):
            checked_name(name)
            expected = f"{self.prefix}/datasets/{dataset_id}/documents/{name}"
            if info["key"] != expected:
                raise ValueError("Document key escaped the dataset")
            data = self._get(expected)
            if digest(data) != info["sha256"]:
                raise ValueError("COS source document checksum mismatch")
            yield name, data, dataset["sources"].get(name, {})

    def load(self):
        with self._lock:
            pointer_bytes = self._get(f"{self.prefix}/current.json", missing_ok=True)
            if pointer_bytes is None:
                return [], {}
            pointer = json.loads(pointer_bytes)
            key = self._index_key(pointer["version"])
            if pointer.get("schema_version") != 1 or pointer.get("index_key") != key:
                raise ValueError("Invalid COS index pointer")
            signature = (pointer["version"], pointer["sha256"])
            if self._cache and self._cache[0] == signature:
                return self._cache[1]
            payload = self._get(key)
            if digest(payload) != pointer["sha256"]:
                raise ValueError("COS knowledge index checksum mismatch")
            snapshot = decode_json(payload)
            if snapshot.get("schema_version") != 1 or snapshot["metadata"].get("version") != pointer["version"]:
                raise ValueError("Invalid COS index snapshot")
            result = snapshot["rows"], snapshot["metadata"]
            self._cache = signature, result
            return result

    def load_version(self, version: str, *, expected_hash: str | None = None):
        """Read a candidate without changing the currently served snapshot."""
        payload = self._get(self._index_key(version))
        if expected_hash and digest(payload) != expected_hash:
            raise ValueError("COS candidate checksum mismatch")
        snapshot = decode_json(payload)
        if snapshot.get("schema_version") != 1 or snapshot["metadata"].get("version") != version:
            raise ValueError("Invalid COS candidate snapshot")
        return snapshot["rows"], snapshot["metadata"]

    def publish(self, rows: list, metadata: dict, dataset_id: str, *, activate: bool = True) -> dict:
        if not rows:
            raise ValueError("Cannot publish an empty knowledge index")
        # Validate that all sources are readable before publishing a new pointer.
        documents = {name for name, _, _ in self.read_documents(dataset_id)}
        if {r["filename"] for r in rows} != documents:
            raise ValueError("Index and source dataset do not match")
        version = uuid.uuid4().hex
        metadata = {**metadata, "dataset_id": dataset_id, "version": version, "storage": "cos"}
        payload = encode_json({"schema_version": 1, "rows": rows, "metadata": metadata})
        compressed = len(payload) > 1_000_000
        if compressed:
            payload = gzip.compress(payload, compresslevel=6, mtime=0)
        self._put(self._index_key(version), payload, compressed=compressed)
        if not activate:
            if digest(self._get(self._index_key(version))) != digest(payload):
                raise ValueError("COS staged index verification failed")
            return {"version": version, "dataset_id": dataset_id, "index_key": self._index_key(version),
                    "sha256": digest(payload), "activated": False}
        return self.activate(version, expected_hash=digest(payload))

    def activate(self, version: str, *, expected_hash: str | None = None) -> dict:
        """Activate a verified index, or roll back to an existing immutable snapshot."""
        key = self._index_key(version)
        payload = self._get(key)
        checksum = digest(payload)
        if expected_hash and checksum != expected_hash:
            raise ValueError("COS index upload verification failed")
        snapshot = decode_json(payload)
        meta = snapshot["metadata"]
        if snapshot.get("schema_version") != 1 or meta["version"] != version or not snapshot["rows"]:
            raise ValueError("Invalid rollback snapshot")
        # Check retained source files before allowing rollback.
        if {name for name, _, _ in self.read_documents(meta["dataset_id"])} != {r["filename"] for r in snapshot["rows"]}:
            raise ValueError("Incomplete source dataset")
        pointer = {"schema_version": 1, "version": version, "index_key": key,
                   "sha256": checksum, "activated_at": datetime.now(timezone.utc).isoformat()}
        # A single-object PUT switches the active version only after all verification succeeds.
        self._put(f"{self.prefix}/current.json", encode_json(pointer))
        with self._lock:
            self._cache = None
        return pointer

    def versions(self) -> list[dict]:
        client = self.client
        marker = ""
        versions = []
        while True:
            response = client.list_objects(Bucket=self.bucket, Prefix=f"{self.prefix}/indexes/",
                                           Marker=marker, MaxKeys=1000)
            objects = response.get("Contents", [])
            for obj in objects:
                version = obj["Key"].rsplit("/", 1)[-1].removesuffix(".json")
                try:
                    checked_version(version)
                except ValueError:
                    continue
                versions.append({"version": version, "last_modified": obj.get("LastModified")})
            if str(response.get("IsTruncated", "false")).lower() != "true":
                break
            next_marker = response.get("NextMarker") or (objects[-1]["Key"] if objects else "")
            if not next_marker or next_marker == marker:
                raise ValueError("Invalid COS listing pagination")
            marker = next_marker
        return versions
