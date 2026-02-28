# Project Goal
Build a Whisper ASR pipeline for long audio files (30 minutes to 1+ hour) that prioritizes maximum throughput by processing the full file with batch-oriented encoding and decoding (real-time performance is not required).

## Project Policy
1. Separate encode and decode stages: encode all chunks first, then run decode.
2. Use continuous dynamic batching with step-level refill (remove finished sequences and immediately backfill new work).
3. Bucket requests by estimated decode length before batching to reduce padding waste.
4. Japanese-only operation: fix language to Japanese (`ja`) and skip language detection.
5. Keep BF16 storage, but use FP32 accumulation for attention score/reduction and final logits.
6. Use paged attention with paged KV cache allocation, stable page mapping, and deterministic free-list behavior.
7. Overlap H2D/D2H transfers with compute using CUDA streams where possible.
8. Use VAD as a scheduling hint (priority/order); do not hard-drop speech segments by default.
9. Beam size is configurable, but must be fixed per run for reproducibility and fair comparison.
10. Add a regression gate: token-level diff plus JA WER/CER on a fixed reference set before merging decoding/kernel changes.
11. Cache encoder outputs for long files so decode-policy experiments can reuse encode results.

## Milestones
1. Milestone 1 (Baseline): Run Whisper `large-v3` on CUDA only, end-to-end, with no optimization applied.
