"""Versioned test samples selected in physical source-row order."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .utils import day_of, norm


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fixed_test_files(paths, directory, fraction, seed):
    """Create once, then reuse a snapshot keyed by input hashes and settings."""
    paths = sorted(map(Path, paths))
    sources = [{"file": p.name, "sha256": _sha256(p), "day": day_of(p)} for p in paths]
    settings = dict(version=1, fraction=fraction, seed=seed, sources=sources,
                    method="PCG64 Bernoulli per file in Parquet physical row order",
                    record_id="source basename:SHA-256:zero-based physical row ordinal")
    if fraction == 1.0:
        settings["method"] = "All eligible records in Parquet physical row order"
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:24]
    directory = Path(directory)
    target = directory / key
    directory.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with TemporaryDirectory(dir=directory) as temp:
            staging = Path(temp) / key
            staging.mkdir()
            counts = []
            for path, source in zip(paths, sources):
                parquet = pq.ParquetFile(path)
                schema = parquet.schema_arrow
                label = next(name for name in schema.names if norm(name) == "label")
                rng = np.random.Generator(np.random.PCG64(
                    np.random.SeedSequence([seed, int(source["sha256"], 16)])))
                prefix = f'{path.name}:{source["sha256"]}:'
                position = selected = eligible = 0
                sample_path = staging / path.name
                with pq.ParquetWriter(sample_path, schema.append(pa.field("record_id", pa.string()))) as writer:
                    for batch in parquet.iter_batches(batch_size=32768):
                        labels = pc.utf8_trim(pc.cast(batch[label], pa.string()), characters=" ")
                        valid = pc.fill_null(pc.not_equal(labels, "Label"), False).to_numpy(zero_copy_only=False)
                        indices = np.flatnonzero(valid if fraction == 1.0 else
                                                 (rng.random(len(batch)) < fraction) & valid)
                        table = pa.Table.from_batches([batch]).take(indices)
                        ids = pa.array([prefix + str(position + int(i)) for i in indices], type=pa.string())
                        writer.write_table(table.append_column("record_id", ids))
                        position += len(batch)
                        selected += len(indices)
                        eligible += int(valid.sum())
                counts.append(dict(day=source["day"], file=path.name, source_rows=position,
                                   eligible_rows=eligible, sample_rows=selected,
                                   schema=str(schema), sample_sha256=_sha256(sample_path)))
            (staging / "manifest.json").write_text(
                json.dumps(dict(settings=settings, counts=counts), indent=2), encoding="utf-8")
            try:
                staging.rename(target)
            except OSError:
                if not target.exists():
                    raise
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest["settings"] != settings:
        raise ValueError(f"Test snapshot manifest mismatch: {target}")
    for row in manifest["counts"]:
        if _sha256(target / row["file"]) != row["sample_sha256"]:
            raise ValueError(f"Test snapshot checksum mismatch: {row['file']}")
    return [target / p.name for p in paths], target / "manifest.json"
