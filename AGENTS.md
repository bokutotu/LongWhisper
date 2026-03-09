## Goal
Build a high-throughput Whisper ASR pipeline for long audio files (30 min to 1+ hour).
Prefer full-file batch processing. Real-time performance is not a goal.

## Rules
- Use C++23
- Assume NVIDIA GPU
- No backward compatibility
- No fallbacks

## Priorities
- Throughput > latency
- Explicit failure > silent recovery
- Simple batch architecture > streaming-oriented design

## Non-Goals
- Real-time ASR
- CPU-only paths
- Compatibility layers

## Commits
- Follow the style of recent full commit messages in this repository
- Do not use one-line-only commit messages
- Include why, what changed, and how it was validated
