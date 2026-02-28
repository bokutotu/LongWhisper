# weights-converter

Convert Whisper `.safetensors` into a packed binary format usable from C++/CUDA.

## Run

```bash
uv run weights-converter \
  --input ../../third_party/whisper-large-v3 \
  --output-dir ../../artifacts/whisper-large-v3-packed
```

## Output Files

- `weights.bin`: concatenated tensor payloads
- `manifest.json`: per-tensor metadata (`name`, `dtype`, `shape`, `offset`, `nbytes`)

## Smoke Test

```bash
uv run weights-converter \
  --input ../../third_party/whisper-large-v3 \
  --output-dir /tmp/packed-test \
  --max-tensors 8
```
