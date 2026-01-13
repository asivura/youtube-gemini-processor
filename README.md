# YouTube Gemini Processor

A universal CLI tool to process any YouTube video using Google's Gemini API. Extracts comprehensive content including transcripts and visual descriptions (slides, diagrams, charts, demonstrations).

## Features

- **Full transcript extraction** with timestamps
- **Visual content descriptions** (slides, diagrams, charts, demos)
- **Multiple analysis modes**: comprehensive, concise, transcript-only
- **Batch processing** support
- **Multiple output formats**: Markdown, JSON
- **Custom prompts** for specialized analysis
- **Run anywhere with uvx** - no installation required

## Quick Start with uvx

No installation needed! Run directly with `uvx`:

```bash
# Set your API key
export GEMINI_API_KEY="your-api-key"

# Run from GitHub (private repo)
uvx --from git+ssh://git@github.com/yourusername/youtube-gemini-processor.git yt-process "https://www.youtube.com/watch?v=VIDEO_ID"

# Or with HTTPS (may require token for private repos)
uvx --from git+https://github.com/yourusername/youtube-gemini-processor.git yt-process "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Installation (Alternative)

```bash
# Install from GitHub
uv pip install git+ssh://git@github.com/yourusername/youtube-gemini-processor.git

# Or clone and install locally
git clone git@github.com:yourusername/youtube-gemini-processor.git
cd youtube-gemini-processor
uv pip install -e .
```

## Prerequisites

1. **Gemini API Key**: Get one from [Google AI Studio](https://aistudio.google.com/app/apikey)

2. **Set the API key**:
   ```bash
   export GEMINI_API_KEY="your-api-key"
   ```

## Usage

### Basic Usage

```bash
# Process a single video (outputs to stdout)
yt-process "https://www.youtube.com/watch?v=VIDEO_ID"

# Save to file
yt-process "https://www.youtube.com/watch?v=VIDEO_ID" -o analysis.md
```

### Analysis Modes

```bash
# Comprehensive (default) - Full transcript + all visuals + summary
yt-process "URL" -m comprehensive

# Concise - Quick summary with key points
yt-process "URL" -m concise

# Transcript only - Just the transcript with visual markers
yt-process "URL" -m transcript
```

### Output Formats

```bash
# Markdown (default)
yt-process "URL" -f markdown -o output.md

# JSON
yt-process "URL" -f json -o output.json
```

### Batch Processing

Create a file with YouTube URLs (one per line):

```text
# urls.txt
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
https://www.youtube.com/watch?v=VIDEO_ID_3
```

Process all videos:

```bash
yt-process --batch urls.txt -o ./output_dir/
```

### Custom Prompts

```bash
# Custom analysis prompt
yt-process "URL" --prompt "Focus on technical implementation details and code examples shown"

# Extract specific information
yt-process "URL" --prompt "List all tools, libraries, and frameworks mentioned"
```

### Model Selection

```bash
# Use different Gemini model
yt-process "URL" --model gemini-2.5-pro

# Available models: gemini-2.5-flash (default), gemini-2.5-pro, gemini-2.0-flash
```

## CLI Options

```
Usage: yt-process [OPTIONS] [URL]

Options:
  -b, --batch PATH                File containing YouTube URLs (one per line)
  -o, --output PATH               Output file or directory (for batch mode)
  -f, --format [markdown|json]    Output format (default: markdown)
  -m, --mode [comprehensive|concise|transcript]
                                  Analysis mode (default: comprehensive)
  -p, --prompt TEXT               Custom prompt (overrides --mode)
  --model TEXT                    Gemini model (default: gemini-2.5-flash)
  --api-key TEXT                  Gemini API key (or set GEMINI_API_KEY)
  -v, --verbose                   Verbose output
  --version                       Show version
  --help                          Show this message and exit
```

## Output Example

### Comprehensive Mode

```markdown
# Video Analysis

**URL**: https://www.youtube.com/watch?v=VIDEO_ID
**Processed**: 2026-01-12T20:30:00
**Model**: gemini-2.5-flash

---

## 1. VIDEO METADATA
- Title: Example Video Title
- Duration: 25:30
- Speaker: John Doe

## 2. FULL TRANSCRIPT
[00:00] John: "Welcome to this presentation..."
[00:15] John: "Today we'll cover..."

## 3. VISUAL CONTENT DESCRIPTIONS

### Visual at [02:30]
**Type**: Slide
**Content**: Architecture diagram showing microservices
**Text on screen**: "System Architecture Overview"

## 4. KEY TOPICS & TIMESTAMPS
- [00:00] Introduction
- [02:30] Architecture Overview

## 5. SUMMARY
...

## 6. KEY TAKEAWAYS
- Key point 1
- Key point 2
```

## API Limits

| Tier | Daily Limit |
|------|-------------|
| Free | 8 hours of YouTube video/day |
| Paid | No limit |

**Note**: Only public YouTube videos are supported.

## Shell Alias (Recommended)

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
alias yt-process='uvx --from git+ssh://git@github.com/yourusername/youtube-gemini-processor.git yt-process'
```

Then use simply:

```bash
yt-process "https://www.youtube.com/watch?v=VIDEO_ID"
```

## License

MIT
