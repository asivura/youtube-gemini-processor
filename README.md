# YouTube Gemini Processor

[![CI](https://github.com/alexandersivura/youtube-gemini-processor/actions/workflows/ci.yml/badge.svg)](https://github.com/alexandersivura/youtube-gemini-processor/actions/workflows/ci.yml)

A CLI tool to process videos using Google's Gemini API. Supports YouTube URLs, local video files, Google Cloud Storage URIs, and the Gemini Files API. Extracts comprehensive content including transcripts, visual descriptions, and structured segment analysis.

## Features

- **Multiple input types**: YouTube URLs, local files, GCS URIs, Files API references
- **Full transcript extraction** with timestamps
- **Visual content descriptions** (slides, diagrams, charts, demos)
- **Segment detection** with optional video splitting via ffmpeg
- **Multiple analysis modes**: comprehensive, concise, transcript-only, segments
- **Video processing controls**: custom FPS, clip ranges, media resolution
- **Batch processing** support
- **Multiple output formats**: Markdown, JSON
- **Custom prompts** for specialized analysis
- **Run anywhere with uvx** - no installation required

## Quick Start with uvx

No installation needed. Run directly with `uvx`:

```bash
# Set your API key
export GEMINI_API_KEY="your-api-key"

# Process a YouTube video
uvx --from git+ssh://git@github.com/asivura/youtube-gemini-processor.git yt-process "https://www.youtube.com/watch?v=VIDEO_ID"

# Process a local video file
uvx --from git+ssh://git@github.com/asivura/youtube-gemini-processor.git yt-process ./video.mp4
```

## Installation

```bash
# Install from GitHub
uv pip install git+ssh://git@github.com/asivura/youtube-gemini-processor.git

# Or clone and install locally
git clone git@github.com:asivura/youtube-gemini-processor.git
cd youtube-gemini-processor
uv pip install -e .
```

## Authentication

The tool supports two authentication methods: Gemini API key and Vertex AI.

### Option 1: Gemini API Key

Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey) and set it as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key"
yt-process "https://www.youtube.com/watch?v=VIDEO_ID"
```

You can also pass it directly:

```bash
yt-process "./video.mp4" --api-key "your-api-key"
```

The tool also checks the `GOOGLE_API_KEY` environment variable as a fallback.

### Option 2: Vertex AI

Vertex AI uses Google Cloud Application Default Credentials (ADC). This is useful when working within a GCP environment or when you need access to GCS-hosted videos.

**Setup:**

```bash
# 1. Install the gcloud CLI (https://cloud.google.com/sdk/docs/install)

# 2. Authenticate with Application Default Credentials
gcloud auth application-default login

# 3. Set your GCP project (pick one method)
export YT_PROCESS_PROJECT="your-gcp-project"
# or pass it via --project flag
```

**Usage:**

```bash
# Use --vertex flag
yt-process "./video.mp4" --vertex --project your-gcp-project

# Or set the environment variable to auto-enable Vertex AI
export GOOGLE_GENAI_USE_VERTEXAI=true
export YT_PROCESS_PROJECT="your-gcp-project"
yt-process "./video.mp4"
```

**Vertex AI environment variables:**

| Variable | Description |
|----------|-------------|
| `YT_PROCESS_PROJECT` | GCP project ID (preferred, tool-specific) |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID (fallback) |
| `YT_PROCESS_LOCATION` | GCP region (preferred, tool-specific) |
| `GOOGLE_CLOUD_LOCATION` | GCP region (fallback, default: `global`) |
| `GOOGLE_GENAI_USE_VERTEXAI` | Set to `true` to auto-enable Vertex AI without `--vertex` flag |

**Note:** The Gemini Files API (used for local file uploads, `--upload-only`, `--list-files`) is not available with Vertex AI. To process local files with Vertex AI, upload them to GCS first (see [GCS URIs](#gcs-uris) below).

### Authentication priority

1. `--api-key` flag
2. `--vertex` flag (Vertex AI with ADC)
3. `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variables
4. `GOOGLE_GENAI_USE_VERTEXAI=true` environment variable

## Input Types

The tool accepts four types of input:

### YouTube URLs

Pass any YouTube URL directly. Supports standard, short, embed, and Shorts URLs:

```bash
yt-process "https://www.youtube.com/watch?v=VIDEO_ID"
yt-process "https://youtu.be/VIDEO_ID"
yt-process "https://www.youtube.com/shorts/VIDEO_ID"
```

Only public YouTube videos are supported.

### Local Video Files

Process video files from your local filesystem. The file is uploaded via the Gemini Files API automatically:

```bash
yt-process ./presentation.mp4
yt-process /path/to/meeting.mov
yt-process ~/Videos/demo.webm
```

Supported formats: `.mp4`, `.mpeg`, `.mov`, `.avi`, `.webm`, `.wmv`, `.flv`, `.mkv`, `.3gp`

### Files API References

Upload a video once and reuse the reference for multiple analyses. Files expire after 48 hours.

```bash
# Upload only - prints the file reference
yt-process ./video.mp4 --upload-only

# Reuse the reference (no re-upload)
yt-process files/abc123 -m comprehensive
yt-process files/abc123 -m segments
yt-process files/abc123 -m transcript

# List uploaded files
yt-process --list-files

# Delete a file
yt-process --delete-file files/abc123
```

### GCS URIs

Process videos stored in Google Cloud Storage. Requires Vertex AI authentication:

```bash
# Upload to GCS first
gcloud storage cp "./video.mp4" gs://your-bucket/

# Process via Vertex AI
yt-process "gs://your-bucket/video.mp4" --vertex --project your-gcp-project
```

## Usage

### Analysis Modes

```bash
# Comprehensive (default) - full transcript, visuals, summary, glossary
yt-process "./video.mp4" -m comprehensive

# Concise - quick summary with key points and slide content
yt-process "./video.mp4" -m concise

# Transcript only - transcript with visual markers
yt-process "./video.mp4" -m transcript

# Segments - identify logical sections with timestamps
yt-process "./video.mp4" -m segments
```

### Output Formats

```bash
# Markdown (default)
yt-process "./video.mp4" -o analysis.md

# JSON
yt-process "./video.mp4" -f json -o analysis.json
```

### Custom Prompts

```bash
# Custom analysis prompt (overrides --mode)
yt-process "./video.mp4" --prompt "Focus on technical implementation details and code examples shown"

# Extract specific information
yt-process "https://www.youtube.com/watch?v=VIDEO_ID" --prompt "List all tools, libraries, and frameworks mentioned"
```

### Model Selection

```bash
yt-process "./video.mp4" --model gemini-3-pro-preview     # Default, best quality
yt-process "./video.mp4" --model gemini-3-flash-preview   # Faster, lower cost
yt-process "./video.mp4" --model gemini-2.5-pro           # Previous gen, high quality
yt-process "./video.mp4" --model gemini-2.5-flash         # Previous gen, fast
yt-process "./video.mp4" --model gemini-2.0-flash         # Oldest supported
```

### Video Processing Options

#### Frame Rate (`--fps`)

Override Gemini's default 1 FPS frame sampling. Higher values capture more detail but increase token usage.

```bash
yt-process ./video.mp4 --fps 2      # More visual detail
yt-process ./video.mp4 --fps 0.5    # Fewer frames, lower cost
```

#### Clip (`--clip`)

Process only a portion of the video:

```bash
yt-process ./video.mp4 --clip 1:30-5:00        # MM:SS format
yt-process ./video.mp4 --clip 90-300            # Raw seconds
yt-process ./video.mp4 --clip 0:01:30-0:05:00   # HH:MM:SS format
```

#### Media Resolution (`--media-resolution`)

Control the resolution at which video frames are processed:

| Value | Tokens/Frame | Use Case |
|-------|-------------|----------|
| `low` | ~66 | Long videos, cost optimization |
| `medium` | (intermediate) | Balanced |
| `high` | ~258 (default) | Detailed visual analysis |

```bash
yt-process ./video.mp4 --media-resolution low
```

#### Combining Options

```bash
# Analyze a specific clip at low resolution with custom FPS
yt-process files/abc123 --clip 0:00-10:00 --fps 0.5 --media-resolution low

# Detailed analysis of a short segment
yt-process ./video.mp4 --clip 5:00-5:30 --fps 5 --media-resolution high
```

### Segments Mode

Identify logical sections of a video with timestamps, speakers, and summaries. Use `--split` to also split the video into separate files via ffmpeg.

```bash
# Identify segments
yt-process "./video.mp4" -m segments

# Identify and split into files (requires ffmpeg)
yt-process "./video.mp4" -m segments --split

# Split YouTube video by chapters (auto-detected from description)
yt-process "https://www.youtube.com/watch?v=VIDEO_ID" --split

# Output segments as JSON
yt-process "./video.mp4" -m segments -f json -o segments.json
```

**Recommended model for segments:** Use `gemini-3-pro-preview` (the default). Smaller models tend to truncate and only cover the first portion of long videos.

### Batch Processing

Create a file with inputs (one per line). Supports YouTube URLs, local files, and GCS URIs:

```text
# inputs.txt
https://www.youtube.com/watch?v=VIDEO_ID_1
./presentation.mp4
gs://my-bucket/meeting.mp4
https://www.youtube.com/watch?v=VIDEO_ID_2
```

```bash
yt-process --batch inputs.txt -o ./output_dir/
```

## CLI Reference

```
Usage: yt-process [OPTIONS] [INPUT]

Options:
  -b, --batch PATH                File containing URLs or file paths (one per line)
  -o, --output PATH               Output file or directory (for batch mode)
  -f, --format [markdown|json]    Output format (default: markdown)
  -m, --mode [comprehensive|concise|transcript|segments]
                                  Analysis mode (default: comprehensive)
  -p, --prompt TEXT               Custom prompt (overrides --mode)
  --model [gemini-3-pro-preview|gemini-3-flash-preview|gemini-2.5-pro|gemini-2.5-flash|gemini-2.0-flash]
                                  Gemini model (default: gemini-3-pro-preview)
  --api-key TEXT                  Gemini API key (or set GEMINI_API_KEY)
  --vertex                       Use Vertex AI authentication
  --project TEXT                  GCP project for Vertex AI
  --location TEXT                 GCP location for Vertex AI
  --fps FLOAT                    Frame sampling rate (default: 1 FPS)
  --clip TEXT                    Process a clip: START-END (e.g., 1:30-5:00)
  --media-resolution [low|medium|high]
                                  Frame resolution (default: high)
  --split                        Split video into segments (segments mode) or by chapters (YouTube URLs)
  --workers INTEGER               Number of parallel workers for --split with YouTube URLs (default: 4)
  --upload-only                  Upload file and print reference without processing
  --list-files                   List files uploaded to the Files API
  --delete-file TEXT             Delete a Files API reference
  -v, --verbose                  Verbose output
  --version                      Show version
  --help                         Show this message and exit
```

## API Limits

| Tier | Daily Limit |
|------|-------------|
| Free | 8 hours of YouTube video/day |
| Paid | No limit |

## Shell Alias (Recommended)

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias yt-process='uvx --from git+ssh://git@github.com/asivura/youtube-gemini-processor.git yt-process'
```

Then use simply:

```bash
yt-process "./video.mp4"
yt-process "https://www.youtube.com/watch?v=VIDEO_ID"
```

## License

MIT
