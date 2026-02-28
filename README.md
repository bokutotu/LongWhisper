# LongWhisper

CUDA-first Whisper experimentation project for long audio throughput.

## Current Status

- Milestone 1 baseline assets are downloaded (`whisper-large-v3` as submodule).
- A `uv` subproject exists for converting `.safetensors` into a simple C++-friendly packed format.

## Model Download (already configured as submodule)

```bash
cd /home/bokutotu/LongWhisper
git submodule update --init --recursive
git -C third_party/whisper-large-v3 lfs pull --include=model.safetensors,config.json,generation_config.json,tokenizer.json,merges.txt,vocab.json,normalizer.json,preprocessor_config.json
```

## Convert Weights For C++/CUDA Runtime

```bash
cd /home/bokutotu/LongWhisper/tools/weights_converter
uv run weights-converter \
  --input ../../third_party/whisper-large-v3 \
  --output-dir ../../artifacts/whisper-large-v3-packed
```

Output:
- `weights.bin`
- `manifest.json`

## Smoke Test

```bash
cd /home/bokutotu/LongWhisper/tools/weights_converter
uv run weights-converter \
  --input ../../third_party/whisper-large-v3 \
  --output-dir /tmp/packed-test \
  --max-tensors 16
```

## Notes

- Weights alone are not enough; tokenizer/config/audio frontend/decode policy are also required.
- See `docs/cpp_cuda_inference_requirements.md` for the full checklist.
