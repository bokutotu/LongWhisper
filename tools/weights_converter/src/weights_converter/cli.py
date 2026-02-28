from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from safetensors import safe_open


@dataclass
class TensorRecord:
    name: str
    dtype: str
    shape: List[int]
    offset: int
    nbytes: int
    source_file: str


def _find_index_file(model_dir: Path) -> Optional[Path]:
    preferred = [
        "model.safetensors.index.json",
        "model.safetensors.index.fp32.json",
    ]
    for filename in preferred:
        candidate = model_dir / filename
        if candidate.exists():
            return candidate
    found = sorted(model_dir.glob("*.safetensors.index*.json"))
    return found[0] if found else None


def _resolve_input(input_path: Path) -> Tuple[str, Path]:
    if input_path.is_file():
        if input_path.suffix == ".json":
            return "index", input_path
        return "single", input_path

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input not found: {input_path}")

    single = input_path / "model.safetensors"
    if single.exists():
        return "single", single

    index_file = _find_index_file(input_path)
    if index_file is not None:
        return "index", index_file

    raise FileNotFoundError(
        f"No model.safetensors or *.safetensors.index*.json under {input_path}"
    )


def _load_index(index_path: Path) -> Dict[str, str]:
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_path}")
    return {str(k): str(v) for k, v in weight_map.items()}


def _iter_tensors(
    input_mode: str,
    source_path: Path,
    max_tensors: Optional[int],
) -> Iterable[Tuple[str, np.ndarray, str]]:
    emitted = 0
    if input_mode == "single":
        with safe_open(str(source_path), framework="np") as f:
            for name in sorted(f.keys()):
                yield name, np.asarray(f.get_tensor(name)), source_path.name
                emitted += 1
                if max_tensors is not None and emitted >= max_tensors:
                    return
        return

    weight_map = _load_index(source_path)
    index_dir = source_path.parent
    current_shard_name: Optional[str] = None
    current_shard = None
    try:
        for name in sorted(weight_map.keys()):
            shard_name = weight_map[name]
            if shard_name != current_shard_name:
                if current_shard is not None:
                    current_shard.close()
                shard_path = index_dir / shard_name
                if not shard_path.exists():
                    raise FileNotFoundError(
                        f"Shard missing for tensor {name}: {shard_path}"
                    )
                current_shard = safe_open(str(shard_path), framework="np")
                current_shard_name = shard_name

            yield name, np.asarray(current_shard.get_tensor(name)), shard_name
            emitted += 1
            if max_tensors is not None and emitted >= max_tensors:
                return
    finally:
        if current_shard is not None:
            current_shard.close()


def _pad_file(file_obj, alignment: int) -> None:
    if alignment <= 1:
        return
    pos = file_obj.tell()
    pad = (alignment - (pos % alignment)) % alignment
    if pad:
        file_obj.write(b"\x00" * pad)


def convert(
    input_path: Path,
    output_dir: Path,
    weights_filename: str,
    manifest_filename: str,
    alignment: int,
    max_tensors: Optional[int],
) -> None:
    if alignment < 1:
        raise ValueError("alignment must be >= 1")

    input_mode, source_path = _resolve_input(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = output_dir / weights_filename
    manifest_path = output_dir / manifest_filename

    records: List[TensorRecord] = []
    converted = 0

    with weights_path.open("wb") as wf:
        for name, tensor, source_file in _iter_tensors(input_mode, source_path, max_tensors):
            tensor = np.ascontiguousarray(tensor)
            _pad_file(wf, alignment)
            offset = wf.tell()
            raw = tensor.tobytes(order="C")
            wf.write(raw)
            records.append(
                TensorRecord(
                    name=name,
                    dtype=str(tensor.dtype),
                    shape=[int(x) for x in tensor.shape],
                    offset=offset,
                    nbytes=len(raw),
                    source_file=source_file,
                )
            )
            converted += 1

    manifest = {
        "format": "longwhisper.packed_weights.v1",
        "source": {
            "input_mode": input_mode,
            "input": str(source_path),
        },
        "weights_file": weights_path.name,
        "alignment": alignment,
        "num_tensors": converted,
        "tensors": [record.__dict__ for record in records],
    }

    with manifest_path.open("w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=True, indent=2)

    print(f"Converted tensors: {converted}")
    print(f"Weights:  {weights_path}")
    print(f"Manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Whisper .safetensors to packed binary + JSON manifest"
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to model.safetensors, index JSON, or model directory",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for packed files",
    )
    parser.add_argument(
        "--weights-filename",
        default="weights.bin",
        help="Output binary filename",
    )
    parser.add_argument(
        "--manifest-filename",
        default="manifest.json",
        help="Output manifest filename",
    )
    parser.add_argument(
        "--alignment",
        type=int,
        default=64,
        help="Byte alignment for each tensor payload in weights.bin",
    )
    parser.add_argument(
        "--max-tensors",
        type=int,
        default=None,
        help="Optional limit for quick smoke testing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        input_path=args.input,
        output_dir=args.output_dir,
        weights_filename=args.weights_filename,
        manifest_filename=args.manifest_filename,
        alignment=args.alignment,
        max_tensors=args.max_tensors,
    )
