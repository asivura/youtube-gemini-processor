# AGENTS.md

This file provides guidance to Codex, Claude Code, and other agents when working with code in this repository.

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
- **Local files** - Files API upload on the Developer API; inline bytes or GCS staging on Vertex; inline base64 on LiteLLM (see below)
- **Files API references** - `files/abc123` references to previously uploaded files (reuse for 48h, Developer API only)
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

### Local Files on Vertex AI

Vertex AI has no Files API — `client.files.upload()` raises `ValueError: This method is only supported in the Gemini Developer client`. `process_local_file()` routes around this automatically:

| File size | Transport | Builder |
|-----------|-----------|---------|
| ≤ `INLINE_MAX_BYTES` (20 MB) | Inline `Part(inline_data=Blob(...))` | `build_inline_media_part()` |
| > 20 MB with `--gcs-bucket` | Staged via `gcloud storage cp`, then `gs://` URI | `upload_to_gcs()` |
| > 20 MB without a bucket | `ClickException` listing all four options | `_build_vertex_local_part()` |

`is_vertex_client()` compares `client.vertexai is True` (not truthiness) so stubs and mocks never take the Vertex path. `_require_developer_api()` guards `--upload-only`, `--list-files`, `--delete-file`, and `files/` inputs with an actionable message.

### GCS Processing (Vertex AI)

A `gs://` URI can always be passed directly as input.

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

**Recommended model**: Use `gemini-3.1-pro-preview` (the default) for segmentation. Smaller models (`gemini-3-flash-preview`, `gemini-3.1-flash-lite`) tend to truncate and only cover the first portion of long videos. The tool auto-detects video duration via ffprobe and injects it into the prompt, but only Pro-tier models reliably cover the full video.

```bash
# Identify segments (uses default gemini-3.1-pro-preview for reliable full-video coverage)
yt-process "./video.mp4" --mode segments

# Identify and split into files
yt-process "./video.mp4" --mode segments --split
```

**Note**: `--split` requires ffmpeg and only works with local files.

## Authentication

Three backends: Gemini API key, Vertex AI (ADC), and a LiteLLM / OpenAI-compatible
endpoint. The LiteLLM backend (`LiteLLMClient` in `cli.py`) targets any
OpenAI-compatible gateway proxying Gemini models via `/chat/completions`. It
duck-types `genai.Client` (`.models.generate_content`) so the `process_*` and
`_call_gemini_and_parse` paths are unchanged. YouTube URLs (and GCS `gs://` URIs)
are sent as an OpenAI `file` content block; `response_schema` maps to
`response_format.json_schema`. Does NOT support the Gemini Files API (local
uploads, `files/*` refs) or `VideoMetadata` (`--fps`/`--clip`/`--media-resolution`,
chapter `--split`) — these are guarded or warn-and-ignored in `main()`.

Priority order:
1. `--litellm` flag (or `LITELLM_BASE_URL` + `LITELLM_API_KEY`)
2. `--api-key` flag
3. `--vertex` flag (Vertex AI with ADC)
4. `GEMINI_API_KEY` or `GOOGLE_API_KEY` env vars
5. `GOOGLE_GENAI_USE_VERTEXAI=true` env var
6. `LITELLM_API_KEY` set with no other auth (auto-enables LiteLLM)

## Models and Pricing

`--model` takes **any** string. `MODEL_PRICING` is pricing data, not an allow-list: an unknown model still runs, and `calculate_cost()` returns `pricing_known=False` with zeroed costs rather than borrowing another model's rates. Adding a model to `MODEL_PRICING` (and `SUGGESTED_MODELS`) only enables cost reporting.

Entry shape: `input`/`output` are per-1M-token rates in USD. Optional `audio_input` overrides `input` for tokens Gemini reports under the AUDIO modality. Optional `long_context_threshold` + `long_input` + `long_output` describe long-prompt pricing.

**Long-context pricing is a cliff, not a graduated bracket.** Google states the rates as "prompts <= 200k tokens" / "prompts > 200k tokens": once the *prompt* crosses the threshold, every token bills at the long rate, and the output rate is selected by the prompt size rather than the output size. `resolve_rates()` owns this. Do not reintroduce proportional bracket-walking — it under-reports any long-context run (a 1M-input / 100k-output `gemini-2.5-pro` call is $4.00, not the $3.25 bracket math produces), and because output caps at 65,536 tokens it would make the high output rate unreachable dead code.

As of August 2026 the frontier Pro model is `gemini-3.1-pro-preview` (the default). **There is no Gemini 3.5 or 3.6 Pro** — the 3.5/3.6 releases are Flash-tier only, and 3.5 Pro has never reached the public API. `gemini-2.5-pro` is carried as the stable fallback because the default is a preview model and preview models get retired on short notice.

### Cost accuracy

`extract_usage()` is the single place token counts are read. Three things it gets right that a naive `candidates_token_count` read does not:

1. **Thinking tokens bill as output.** `thoughts_token_count` is reported separately by the API but charged at the output rate. It is folded into `UsageStats.output_tokens` and broken out in `thoughts_tokens`. On Gemini 3 models thinking often dominates: a short call can be 9 visible tokens against 207 reasoning tokens.
2. **Audio input has its own rate** on several models. `prompt_tokens_details` gives the per-modality split.
3. **Non-integer fields coerce to 0** via `_as_int()` rather than propagating into format strings.

## Exit Codes

`main()` raises `SystemExit(1)` if any input failed and `SystemExit(2)` if all succeeded but output was truncated. Never make failures exit 0 — an error document written to disk with a success code is undetectable from a script and was the tool's worst automation bug.

## Truncation

`_call_gemini_and_parse()` reads `finish_reason`. `MAX_TOKENS` sets `analysis.truncated`, warns on stderr, and adds a banner to the document. `response.text` is `Optional[str]` and is `None` when a response is blocked or spends its whole budget on reasoning; that case raises a `ClickException` naming the finish reason instead of a bare `TypeError` from the regexes.

## Prompts

`PROMPTS` (video) and `AUDIO_PROMPTS` (audio), selected by `select_prompt(mode, kind)`. Audio variants exist because the video prompts request "Visual Content" sections and `[SLIDE: ...]` markers, which lead a model to invent slides for an audio-only file.

Every built-in prompt carries a `{duration_line}` placeholder that **must** be filled via `.format()` before sending — an unformatted template ships the literal string `{duration_line}` to the model. Custom `--prompt` text is never formatted, since user text may contain braces.
