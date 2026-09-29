"""續跑狀態：設定摘要、輸入雜湊與既有產物必須一致。"""
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def load_resume_rows(path, fields, protocol_id, key_fields):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"{path}: CSV schema 不符或缺少續跑摘要；請指定新的輸出，不改寫舊表")
        rows = list(reader)
    seen = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"{path}: CSV 列不完整")
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            raise ValueError(f"{path}: 重複續跑鍵 {key}")
        seen.add(key)
        if row["protocol_id"] != protocol_id:
            raise ValueError(f"{path}: 協定不符 {key}；請指定新的 CSV 與影像輸出")
        for column in ("png", "input_png", "reference_png"):
            if column not in row:
                continue
            artifact = Path(row[column])
            if not artifact.is_file():
                raise FileNotFoundError(f"{path}: {key} 缺少 {column}: {artifact}；保留原始列")
            digest_column = {"input_png": "input_sha256", "reference_png": "reference_sha256"}.get(column)
            if digest_column and file_digest(artifact) != row[digest_column]:
                raise ValueError(f"{path}: {key} 的 {column} 內容已變更")
    return rows


def write_rows_atomic(path, fields, rows):
    """單一 writer 更新續跑表，中斷不得留下半張 CSV。"""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
