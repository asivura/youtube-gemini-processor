# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video Processor CLI for processing videos using Google's Gemini API. Supports YouTube URLs and local video files. Extracts transcripts, visual descriptions, and detailed analysis in multiple formats.

## Commands

```bash
# Install dependencies
uv pip install -e .

# Run the tool (YouTube URL or local file)
yt-process <youtube-url-or-file>

# Examples
yt-process "https://www.youtube.com/watch?v=VIDEO_ID"
yt-process ./video.mp4

# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Run with verbose output
yt-process -v ./video.mp4
```

## Architecture

Single-module CLI application in `src/youtube_gemini_processor/cli.py`:

| Component | Purpose |
|-----------|---------|
| `UsageStats` | Dataclass for token usage and cost tracking |
| `VideoAnalysis` | Dataclass for structured analysis output |
| `get_gemini_client()` | Initialize Gemini API client (API key or Vertex AI) |
| `is_local_file()` | Detect if input is a local file path |
| `is_files_api_ref()` | Detect if input is a Files API reference |
| `validate_youtube_url()` | Parse and normalize YouTube URL formats |
| `process_video()` | Process YouTube videos via URL |
| `process_local_file()` | Process local files via Files API upload |
| `process_files_api_ref()` | Process using existing Files API reference (no upload) |
| `parse_timestamp_to_seconds()` | Parse SS, MM:SS, HH:MM:SS to `"{seconds}s"` |
| `parse_clip_range()` | Parse `"START-END"` clip range string |
| `get_media_mime_type()` | Resolve MIME type and kind (`video`/`audio`) for a media file |
| `build_media_part()` | Build media Part; VideoMetadata attached only for video (or audio clip offsets) |
| `build_generate_config()` | Build GenerateContentConfig with optional media resolution |
| `_call_gemini_and_parse()` | Shared helper: call Gemini API and populate VideoAnalysis |
| `_format_duration()` | Format seconds as HH:MM:SS |
| `_normalize_timestamp_to_hhmmss()` | Normalize MM:SS or HH:MM:SS timestamps |
| `calculate_cost()` | Token usage cost calculation with model pricing |
| `format_output_*()` | Output formatters (markdown, json) |
| `_handle_file_management()` | Handle `--list-files` and `--delete-file` operations |
| `_handle_upload_only()` | Handle `--upload-only` mode |
| `_handle_chapter_splitting()` | Handle YouTube chapter-based `--split` |
| `_handle_output()` | Route formatted output to file or stdout |

## Input Types

- **YouTube URLs** - Passed directly to Gemini via `file_uri` (video only)
- **Local files** - Uploaded via Gemini Files API, then processed (API key only, not Vertex AI)
- **Files API references** - `files/abc123` references to previously uploaded files (reuse for 48h)
- **GCS URIs** - `gs://` paths processed directly via Vertex AI

Supported video formats: `.mp4`, `.mpeg`, `.mov`, `.avi`, `.webm`, `.wmv`, `.flv`, `.mkv`, `.3gp`
Supported audio formats: `.mp3`, `.m4a`, `.wav`, `.flac`, `.ogg`, `.aac`, `.aiff`, `.aif`

`--fps` and `--media-resolution` are video-only and error out if combined with audio input. `--clip` works for both.

### Files API Reuse

Upload a video once and reuse the reference for multiple analyses (expires after 48 hours):

```bash
# Upload only — prints the file reference
yt-process ./video.mp4 --upload-only

# Reuse the reference (no re-upload, no re-processing)
yt-process files/abc123 -m comprehensive
yt-process files/abc123 -m segments
yt-process files/abc123 -m transcript

# List uploaded files
yt-process --list-files

# Delete a file
yt-process --delete-file files/abc123
```

### GCS Processing (Vertex AI)

For Vertex AI processing (when `GOOGLE_GENAI_USE_VERTEXAI=true`), local files must be uploaded to GCS first since the Files API is not supported with Vertex AI.

```bash
# Upload and process via Vertex AI
gcloud storage cp "./video.mp4" gs://your-bucket/
uv run yt-process "gs://your-bucket/video.mp4" \
  -v --vertex --project your-gcp-project
```

## Video Processing Options

Control how Gemini processes video frames using `--fps`, `--clip`, and `--media-resolution`. These options work with all input types (local files, Files API refs, GCS URIs, YouTube URLs).

### Frame Rate (`--fps`)

Override Gemini's default 1 FPS frame sampling. Higher values capture more detail but increase token usage.

```bash
# Sample at 2 FPS (more visual detail)
yt-process ./video.mp4 --fps 2

# Sample at 0.5 FPS (fewer frames, lower cost)
yt-process ./video.mp4 --fps 0.5

# Works with Files API references too
yt-process files/abc123 --fps 2
```

**Token impact**: Default 1 FPS = ~300 tokens/sec. Higher FPS increases proportionally.

### Clip (`--clip`)

Process only a portion of the video. Accepts `START-END` in multiple timestamp formats.

```bash
# Process 1:30 to 5:00 (MM:SS format)
yt-process ./video.mp4 --clip 1:30-5:00

# Raw seconds
yt-process ./video.mp4 --clip 90-300

# HH:MM:SS format
yt-process ./video.mp4 --clip 0:01:30-0:05:00

# With seconds suffix
yt-process ./video.mp4 --clip 90s-300s
```

### Media Resolution (`--media-resolution`)

Control the resolution at which video frames are processed. Lower resolution saves tokens for long videos.

| Value | Tokens/Frame | Use Case |
|-------|-------------|----------|
| `low` | ~66 | Long videos, cost optimization |
| `medium` | (intermediate) | Balanced |
| `high` | ~258 (default) | Detailed visual analysis |

```bash
# Low resolution for a long lecture
yt-process ./video.mp4 --media-resolution low

# High resolution for detailed visual inspection
yt-process ./video.mp4 --media-resolution high
```

### Combining Options

```bash
# Analyze a specific clip at low resolution with custom FPS
yt-process files/abc123 --clip 0:00-10:00 --fps 0.5 --media-resolution low

# Detailed analysis of a short segment
yt-process ./video.mp4 --clip 5:00-5:30 --fps 5 --media-resolution high
```

## Analysis Modes

Four hardcoded prompts in `cli.py`:
- `DEFAULT_PROMPT` (comprehensive) - Full transcript, visuals, summary, glossary
- `CONCISE_PROMPT` - Quick summary with key points
- `TRANSCRIPT_ONLY_PROMPT` - Transcript with visual markers
- `SEGMENTS_PROMPT` - Identify logical sections with timestamps, titles, speakers, and summaries (uses `response_schema` for guaranteed valid JSON output)

### Segments Mode

The `--mode segments` option uses Gemini to identify logical sections of a video. It returns structured JSON with segment boundaries. Use `--split` to also split the video into separate files via ffmpeg.

**Recommended model**: Use `gemini-3-pro-preview` for segmentation. Smaller models (`gemini-3-flash-preview`, `gemini-2.5-flash`) tend to truncate and only cover the first portion of long videos. The tool auto-detects video duration via ffprobe and injects it into the prompt, but only `gemini-3-pro-preview` reliably covers the full video.

```bash
# Identify segments (use gemini-3-pro for reliable full-video coverage)
yt-process "./video.mp4" --mode segments --model gemini-3-pro-preview

# Identify and split into files
yt-process "./video.mp4" --mode segments --split --model gemini-3-pro-preview
```

**Note**: `--split` requires ffmpeg and only works with local files.

## Authentication

Priority order:
1. `--api-key` flag
2. `--vertex` flag (Vertex AI with ADC)
3. `GEMINI_API_KEY` or `GOOGLE_API_KEY` env vars
4. `GOOGLE_GENAI_USE_VERTEXAI=true` env var

## Model Pricing

Hardcoded in `MODEL_PRICING` dict. When adding new models, update this dictionary with input/output costs per 1M tokens.
