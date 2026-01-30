# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YouTube Gemini Processor is a CLI tool for processing YouTube videos using Google's Gemini API. It extracts transcripts, visual descriptions, and detailed analysis in multiple formats.

## Commands

```bash
# Install dependencies
uv pip install -e .

# Run the tool
yt-process <youtube-url>

# Lint
uv run ruff check src/

# Format
uv run ruff format src/

# Run with verbose output
yt-process -v <youtube-url>
```

## Architecture

Single-module CLI application in `src/youtube_gemini_processor/cli.py`:

| Component | Purpose |
|-----------|---------|
| `UsageStats` | Dataclass for token usage and cost tracking |
| `VideoAnalysis` | Dataclass for structured analysis output |
| `get_gemini_client()` | Initialize Gemini API client (API key or Vertex AI) |
| `validate_youtube_url()` | Parse and normalize YouTube URL formats |
| `process_video()` | Main processing function using Gemini API |
| `calculate_cost()` | Token usage cost calculation with model pricing |
| `format_output_*()` | Output formatters (markdown, json) |

## Analysis Modes

Three hardcoded prompts in `cli.py`:
- `DEFAULT_PROMPT` (comprehensive) - Full transcript, visuals, summary, glossary
- `CONCISE_PROMPT` - Quick summary with key points
- `TRANSCRIPT_ONLY_PROMPT` - Transcript with visual markers

## Authentication

Priority order:
1. `--api-key` flag
2. `--vertex` flag (Vertex AI with ADC)
3. `GEMINI_API_KEY` or `GOOGLE_API_KEY` env vars
4. `GOOGLE_GENAI_USE_VERTEXAI=true` env var

## Model Pricing

Hardcoded in `MODEL_PRICING` dict. When adding new models, update this dictionary with input/output costs per 1M tokens.
