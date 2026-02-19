#!/usr/bin/env python3
"""
Video Processor CLI

A universal CLI tool to process videos using Google's Gemini API.
Supports YouTube URLs and local video files.
Extracts comprehensive content including transcripts and visual descriptions.

Usage:
    yt-process <youtube_url_or_file> [options]
    yt-process --batch <file_with_urls_or_paths> [options]

Examples:
    # Process a YouTube video
    yt-process "https://www.youtube.com/watch?v=VIDEO_ID"

    # Process a local video file
    yt-process ./video.mp4

    # Process with custom output
    yt-process "https://youtube.com/watch?v=XYZ" -o output.md

    # Process multiple videos from a file
    yt-process --batch urls.txt -o ./output_dir/

    # Get JSON output
    yt-process ./presentation.mp4 --format json

    # Custom analysis prompt
    yt-process ./video.mp4 --prompt "Focus on technical details"
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    pass


@dataclass
class UsageStats:
    """Token usage and cost statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0


@dataclass
class VideoAnalysis:
    """Structured output from video analysis."""

    url: str
    title: str = ""
    duration: str = ""
    transcript: str = ""
    visual_descriptions: list[dict] = field(default_factory=list)
    key_topics: list[str] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""
    processed_at: str = ""
    model: str = ""
    error: str | None = None
    usage: UsageStats | None = None


# Default comprehensive analysis prompt - optimized for maximum detail
DEFAULT_PROMPT = """You are an expert content analyst. Analyze this video with extreme thoroughness and produce a comprehensive markdown document.

# INSTRUCTIONS
- Watch the ENTIRE video carefully
- Capture EVERY piece of information: spoken words, visuals, text on screen
- Be extremely detailed - this document should allow someone to fully understand the video without watching it
- Use proper markdown formatting throughout

# OUTPUT FORMAT

## Video Information

| Field | Value |
|-------|-------|
| **Title** | [Full video title] |
| **Duration** | [HH:MM:SS] |
| **Speaker(s)** | [Name(s) and role(s)/title(s) if mentioned] |
| **Organization** | [Company/org if mentioned] |
| **Topic** | [Main subject] |

---

## Executive Summary

[3-5 paragraph comprehensive summary covering:
- What the video is about
- Who it's for
- Main arguments/points made
- Key conclusions]

---

## Table of Contents

[Create a clickable table of contents with timestamps for each major section]

| Time | Section |
|------|---------|
| [00:00] | Section name |
| [MM:SS] | Section name |

---

## Detailed Content

### [Section Title] [MM:SS - MM:SS]

[For EACH section of the video, provide:]

**Summary**: [2-3 sentence summary of this section]

**Key Points**:
- Point 1
- Point 2
- Point 3

**Transcript Excerpt**:
> "[Important quotes from this section with timestamps]"

**Visual Content**:
- **[MM:SS]** - [Type: Slide/Chart/Diagram/Demo/Code]
  - Description: [Detailed description of what's shown]
  - Text on screen: [ALL text visible, verbatim]
  - Key data: [Any numbers, statistics, or data shown]

[Repeat for each major section]

---

## All Visual Content with Speaker Notes (Comprehensive)

[List EVERY visual element shown in chronological order, with detailed speaker notes]

### Slide/Visual 1 - [MM:SS - MM:SS]
- **Type**: [Slide/Diagram/Chart/Code/Demo/Screenshot]
- **Title/Header**: [Text shown as title]
- **Content**: [Describe everything shown]
- **Full Text**: [Transcribe ALL text visible on this visual]
- **Data/Numbers**: [Any statistics, percentages, figures]
- **Speaker Notes**: [Detailed notes of what the speaker says while this visual is shown. Include key explanations, examples, stories, and insights. This should capture the speaker's commentary that accompanies this visual - not just a summary but detailed notes of their points.]

[Continue for ALL visuals]

### Introduction (Before First Slide) - [00:00 - MM:SS]
- **Speaker Notes**: [What the speaker says before the first slide appears]

### Transitions & Non-Slide Content
[Capture any important content spoken between slides or when no slides are shown]

---

## Full Transcript

[Provide the COMPLETE transcript with timestamps every 30 seconds minimum]

**[00:00]** Speaker: "..."

**[00:30]** Speaker: "..."

[Continue for entire video - DO NOT summarize or skip parts]

---

## Key Takeaways

### Main Lessons
1. [Lesson 1 - with brief explanation]
2. [Lesson 2 - with brief explanation]
3. [Continue...]

### Actionable Advice
- [ ] Action item 1
- [ ] Action item 2
- [ ] Continue...

### Memorable Quotes
> "[Quote 1]" - [Speaker, MM:SS]

> "[Quote 2]" - [Speaker, MM:SS]

---

## Resources Mentioned

| Resource | Type | Link/Reference |
|----------|------|----------------|
| [Name] | [Book/Tool/Website/etc] | [URL if shown] |

---

## Glossary

| Term | Definition |
|------|------------|
| [Term used] | [Explanation as given in video] |

---

## Related Topics

- [Topics mentioned that viewers might want to explore further]

---

*Analysis generated by Gemini Video Processor*

# CRITICAL REMINDERS
- Include EVERY slide and visual - do not skip any
- Transcribe ALL text shown on screen
- Capture the FULL transcript, not a summary
- Include DETAILED SPEAKER NOTES for each slide - what the speaker explains, examples they give, stories they tell
- Capture content spoken BEFORE the first slide and BETWEEN slides
- Be extremely thorough - more detail is better
- Use proper markdown tables, headers, and formatting"""


CONCISE_PROMPT = """Analyze this video. Provide:

1. **Title & Duration**
2. **Summary** (2-3 paragraphs)
3. **Key Topics** with timestamps [MM:SS]
4. **Main Takeaways** (bullet points)
5. **Slides & Visual Content** - For EACH slide or visual shown in the video:
   - Timestamp [MM:SS - MM:SS] (when slide appears and disappears)
   - Slide title/header (if present)
   - ALL text content on the slide (bullet points, lists, etc.)
   - Any diagrams, charts, or images with descriptions
   - Key data points or statistics shown
   - **Speaker Notes**: What the speaker says while this slide is shown (key quotes and explanations, not verbatim transcript but detailed notes capturing the main points, examples, and insights shared)

Be comprehensive. Capture ALL slide content verbatim and detailed speaker commentary for each slide."""


TRANSCRIPT_ONLY_PROMPT = """Provide a complete transcript of this video.

Format with timestamps:
[MM:SS] "Spoken text..."

Include speaker identification if multiple speakers.
Also note any significant visual content shown (slides, demos) in brackets like:
[MM:SS] [SLIDE: Title of slide or description]
[MM:SS] [DEMO: What is being demonstrated]"""


SEGMENTS_PROMPT = """You are an expert video analyst. Watch this ENTIRE video carefully from start to finish and identify all logical sections/segments.
{duration_line}
# INSTRUCTIONS
- Identify every major topic change, speaker transition, or agenda item boundary
- Focus on semantic/content transitions, not minor pauses
- Each segment should represent a coherent topic or agenda item
- Provide accurate timestamps in HH:MM:SS format

# OUTPUT FORMAT

Return ONLY a JSON array (no markdown fencing, no other text) with this structure:

[
  {{
    "segment_number": 1,
    "start_time": "00:00:00",
    "end_time": "00:05:30",
    "title": "Opening Remarks",
    "speaker": "Speaker Name (if identifiable)",
    "summary": "Brief 1-2 sentence summary of what happens in this segment"
  }},
  {{
    "segment_number": 2,
    "start_time": "00:05:30",
    "end_time": "00:15:00",
    "title": "Product Update",
    "speaker": "Speaker Name",
    "summary": "Brief summary"
  }}
]

# CRITICAL RULES
- Return ONLY the JSON array, no other text or markdown
- Cover the ENTIRE video from start to finish with no gaps
- The LAST segment's end_time MUST match the video's total duration
- Each segment's start_time should equal the previous segment's end_time
- Use HH:MM:SS format for all timestamps
- Be specific with segment titles (not generic like "Section 1")
- Include speaker name if identifiable, otherwise use "Unknown" or a description"""


PROMPTS = {
    "comprehensive": DEFAULT_PROMPT,
    "concise": CONCISE_PROMPT,
    "transcript": TRANSCRIPT_ONLY_PROMPT,
    "segments": SEGMENTS_PROMPT,
}

# Pricing per 1M tokens (as of Jan 2025)
# https://ai.google.dev/pricing
MODEL_PRICING = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},  # Per 1M tokens
    "gemini-3-pro-preview": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}

# Maximum output tokens per model
MODEL_MAX_OUTPUT_TOKENS = {
    "gemini-3-flash-preview": 65536,
    "gemini-3-pro-preview": 65536,
    "gemini-2.5-flash": 65536,
    "gemini-2.5-pro": 65536,
    "gemini-2.0-flash": 8192,
}


def get_max_output_tokens(model: str) -> int:
    """Get maximum output tokens for a model."""
    return MODEL_MAX_OUTPUT_TOKENS.get(model, 8192)


# JSON Schema for segments mode - enforces structured output from Gemini
SEGMENTS_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "segment_number": {"type": "INTEGER"},
            "start_time": {"type": "STRING"},
            "end_time": {"type": "STRING"},
            "title": {"type": "STRING"},
            "speaker": {"type": "STRING"},
            "summary": {"type": "STRING"},
        },
        "required": [
            "segment_number",
            "start_time",
            "end_time",
            "title",
            "speaker",
            "summary",
        ],
    },
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> UsageStats:
    """Calculate usage cost based on model and token counts."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gemini-3-pro-preview"])

    # Convert to cost (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
    )


def get_gemini_client(
    api_key: str | None = None,
    use_vertex: bool = False,
    project: str | None = None,
    location: str | None = None,
):
    """
    Initialize Gemini client with API key or Vertex AI authentication.

    Authentication methods (in order of priority):
    1. Explicit API key (--api-key flag)
    2. Vertex AI with ADC (--vertex flag) - uses gcloud auth
    3. Environment variables (GEMINI_API_KEY or GOOGLE_API_KEY)
    4. Auto-detect Vertex AI if GOOGLE_GENAI_USE_VERTEXAI=true
    """
    from google import genai

    # Check if Vertex AI mode is requested or auto-detected
    vertex_env = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
    use_vertex = use_vertex or vertex_env

    if use_vertex:
        # Vertex AI authentication using Application Default Credentials
        # Requires: gcloud auth application-default login
        gcp_project = (
            project
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT")
            or os.environ.get("CLOUDSDK_CORE_PROJECT")
        )
        gcp_location = (
            location
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or os.environ.get("CLOUDSDK_COMPUTE_REGION")
            or "us-central1"
        )

        if not gcp_project:
            raise click.ClickException(
                "Vertex AI requires a GCP project. Set GOOGLE_CLOUD_PROJECT "
                "environment variable or pass --project"
            )

        click.echo(
            f"Using Vertex AI (project: {gcp_project}, location: {gcp_location})",
            err=True,
        )

        return genai.Client(
            vertexai=True,
            project=gcp_project,
            location=gcp_location,
        )

    # API key authentication
    key = (
        api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    )
    if not key:
        raise click.ClickException(
            "Authentication required. Choose one:\n"
            "  1. API key: Set GEMINI_API_KEY env var or pass --api-key\n"
            "  2. Vertex AI: Pass --vertex flag (requires gcloud auth application-default login)\n"
            "\nGet an API key at: https://aistudio.google.com/app/apikey"
        )
    return genai.Client(api_key=key)


def validate_youtube_url(url: str) -> str:
    """Validate and normalize YouTube URL."""
    patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            return f"https://www.youtube.com/watch?v={video_id}"

    raise click.ClickException(f"Invalid YouTube URL: {url}")


def extract_video_id(url: str) -> str | None:
    """Extract video ID from YouTube URL."""
    match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


# Supported video MIME types
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".webm": "video/webm",
    ".wmv": "video/x-ms-wmv",
    ".flv": "video/x-flv",
    ".mkv": "video/x-matroska",
    ".3gp": "video/3gpp",
}


def is_local_file(input_path: str) -> bool:
    """Check if input is a local file path (exists on disk)."""
    path = Path(input_path)
    return path.exists() and path.is_file()


def is_youtube_url(url: str) -> bool:
    """Check if input looks like a YouTube URL."""
    youtube_patterns = [
        r"(?:https?://)?(?:www\.)?youtube\.com",
        r"(?:https?://)?(?:www\.)?youtu\.be",
    ]
    return any(re.search(pattern, url) for pattern in youtube_patterns)


def is_gcs_uri(uri: str) -> bool:
    """Check if input is a Google Cloud Storage URI."""
    return uri.startswith("gs://")


def get_video_duration(file_path: str) -> str | None:
    """Get video duration in HH:MM:SS format using ffprobe.

    Returns None if ffprobe is not available or fails.
    """
    if not shutil.which("ffprobe"):
        return None
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                file_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        seconds = float(result.stdout.strip())
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    except (ValueError, OSError):
        return None


def get_video_mime_type(file_path: Path) -> str:
    """Get MIME type for a video file based on extension."""
    ext = file_path.suffix.lower()
    mime_type = VIDEO_MIME_TYPES.get(ext)
    if not mime_type:
        raise click.ClickException(
            f"Unsupported video format: {ext}\n"
            f"Supported formats: {', '.join(VIDEO_MIME_TYPES.keys())}"
        )
    return mime_type


def process_local_file(
    client,
    file_path: str,
    prompt: str,
    model: str = "gemini-3-pro-preview",
    verbose: bool = False,
    response_schema: dict | None = None,
) -> VideoAnalysis:
    """Process a local video file with Gemini API using Files API."""
    from google.genai import types

    path = Path(file_path).resolve()

    analysis = VideoAnalysis(
        url=str(path),
        processed_at=datetime.now().isoformat(),
        model=model,
    )

    try:
        mime_type = get_video_mime_type(path)
        file_size_mb = path.stat().st_size / (1024 * 1024)

        if verbose:
            click.echo(f"  Uploading {path.name} ({file_size_mb:.1f} MB)...", err=True)

        # Upload file using Files API
        uploaded_file = client.files.upload(file=str(path))

        if verbose:
            click.echo(f"  File uploaded: {uploaded_file.name}", err=True)

        # Wait for file to be processed
        import time

        while uploaded_file.state.name == "PROCESSING":
            if verbose:
                click.echo("  Waiting for file processing...", err=True)
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise click.ClickException(f"File processing failed: {uploaded_file.name}")

        # Build config
        config_kwargs: dict = {
            "max_output_tokens": get_max_output_tokens(model),
        }
        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        # Generate content using the uploaded file
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            file_data=types.FileData(
                                file_uri=uploaded_file.uri,
                                mime_type=mime_type,
                            )
                        ),
                        types.Part(text=prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(**config_kwargs),
        )

        analysis.raw_response = response.text

        # Extract usage stats from response
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            analysis.usage = calculate_cost(model, input_tokens, output_tokens)

        # Try to extract title from response
        title_match = re.search(
            r"(?:title|video)[:\s]*[\"']?([^\n\"']+)[\"']?",
            response.text,
            re.IGNORECASE,
        )
        if title_match:
            analysis.title = title_match.group(1).strip()
        else:
            analysis.title = path.stem  # Use filename as fallback

        # Extract summary if present
        summary_match = re.search(
            r"(?:## (?:5\. )?summary|summary:)\s*\n(.*?)(?=\n## |\n\*\*|\Z)",
            response.text,
            re.IGNORECASE | re.DOTALL,
        )
        if summary_match:
            analysis.summary = summary_match.group(1).strip()

    except Exception as e:
        analysis.error = str(e)

    return analysis


def process_gcs_uri(
    client,
    gcs_uri: str,
    prompt: str,
    model: str = "gemini-3-pro-preview",
    verbose: bool = False,
    response_schema: dict | None = None,
) -> VideoAnalysis:
    """Process a video from Google Cloud Storage with Gemini API."""
    from google.genai import types

    # Extract filename from GCS URI for display
    filename = gcs_uri.split("/")[-1]

    # Determine MIME type from extension
    ext = Path(filename).suffix.lower()
    mime_type = VIDEO_MIME_TYPES.get(ext, "video/mp4")

    analysis = VideoAnalysis(
        url=gcs_uri,
        processed_at=datetime.now().isoformat(),
        model=model,
    )

    try:
        if verbose:
            click.echo(f"  Processing GCS file: {filename}", err=True)

        # Build config
        config_kwargs: dict = {
            "max_output_tokens": get_max_output_tokens(model),
        }
        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            file_data=types.FileData(
                                file_uri=gcs_uri,
                                mime_type=mime_type,
                            )
                        ),
                        types.Part(text=prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(**config_kwargs),
        )

        analysis.raw_response = response.text

        # Extract usage stats from response
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            analysis.usage = calculate_cost(model, input_tokens, output_tokens)

        # Try to extract title from response
        title_match = re.search(
            r"(?:title|video)[:\s]*[\"']?([^\n\"']+)[\"']?",
            response.text,
            re.IGNORECASE,
        )
        if title_match:
            analysis.title = title_match.group(1).strip()
        else:
            analysis.title = Path(filename).stem  # Use filename as fallback

        # Extract summary if present
        summary_match = re.search(
            r"(?:## (?:5\. )?summary|summary:)\s*\n(.*?)(?=\n## |\n\*\*|\Z)",
            response.text,
            re.IGNORECASE | re.DOTALL,
        )
        if summary_match:
            analysis.summary = summary_match.group(1).strip()

    except Exception as e:
        analysis.error = str(e)

    return analysis


def process_video(
    client,
    url: str,
    prompt: str,
    model: str = "gemini-3-pro-preview",
    response_schema: dict | None = None,
) -> VideoAnalysis:
    """Process a single YouTube video with Gemini API."""
    from google.genai import types

    normalized_url = validate_youtube_url(url)

    analysis = VideoAnalysis(
        url=normalized_url,
        processed_at=datetime.now().isoformat(),
        model=model,
    )

    try:
        # Build config
        config_kwargs: dict = {
            "max_output_tokens": get_max_output_tokens(model),
        }
        if response_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            file_data=types.FileData(
                                file_uri=normalized_url,
                                mime_type="video/mp4",  # Required for Vertex AI
                            )
                        ),
                        types.Part(text=prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(**config_kwargs),
        )

        analysis.raw_response = response.text

        # Extract usage stats from response
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            output_tokens = getattr(usage, "candidates_token_count", 0) or 0
            analysis.usage = calculate_cost(model, input_tokens, output_tokens)

        # Try to extract title from response
        title_match = re.search(
            r"(?:title|video)[:\s]*[\"']?([^\n\"']+)[\"']?",
            response.text,
            re.IGNORECASE,
        )
        if title_match:
            analysis.title = title_match.group(1).strip()

        # Extract summary if present
        summary_match = re.search(
            r"(?:## (?:5\. )?summary|summary:)\s*\n(.*?)(?=\n## |\n\*\*|\Z)",
            response.text,
            re.IGNORECASE | re.DOTALL,
        )
        if summary_match:
            analysis.summary = summary_match.group(1).strip()

    except Exception as e:
        analysis.error = str(e)

    return analysis


def parse_segments(raw_response: str) -> list[dict]:
    """Parse segment data from Gemini's JSON response.

    Handles responses that may contain markdown fencing or extra text around the JSON.
    """
    text = raw_response.strip()

    # Strip markdown code fences if present
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    # Try to find JSON array in the text
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start != -1 and bracket_end != -1:
        text = text[bracket_start : bracket_end + 1]

    try:
        segments = json.loads(text)
    except json.JSONDecodeError as e:
        raise click.ClickException(
            f"Failed to parse segments JSON from response: {e}"
        ) from None

    if not isinstance(segments, list) or not segments:
        raise click.ClickException("Response did not contain a valid segments array")

    return segments


def format_segments_markdown(analysis: VideoAnalysis, segments: list[dict]) -> str:
    """Format segment analysis as a readable markdown document."""
    if analysis.error:
        return f"# Error Processing Video\n\n**URL**: {analysis.url}\n\n**Error**: {analysis.error}\n"

    usage_section = ""
    if analysis.usage:
        u = analysis.usage
        usage_section = f"""
**Usage**: {u.input_tokens:,} input + {u.output_tokens:,} output = {u.total_tokens:,} tokens
**Cost**: ${u.total_cost:.6f} (${u.input_cost:.6f} input + ${u.output_cost:.6f} output)
"""

    lines = [
        "# Video Segments Analysis",
        "",
        f"**Source**: {analysis.url}",
        f"**Processed**: {analysis.processed_at}",
        f"**Model**: {analysis.model}{usage_section}",
        "",
        "---",
        "",
        "## Segments",
        "",
        "| # | Start | End | Title | Speaker |",
        "|---|-------|-----|-------|---------|",
    ]

    for seg in segments:
        num = seg.get("segment_number", "")
        start = seg.get("start_time", "")
        end = seg.get("end_time", "")
        title = seg.get("title", "")
        speaker = seg.get("speaker", "")
        lines.append(f"| {num} | {start} | {end} | {title} | {speaker} |")

    lines.extend(["", "---", "", "## Segment Details", ""])

    for seg in segments:
        num = seg.get("segment_number", "")
        title = seg.get("title", "Untitled")
        start = seg.get("start_time", "")
        end = seg.get("end_time", "")
        speaker = seg.get("speaker", "")
        summary = seg.get("summary", "")

        lines.append(f"### {num}. {title} [{start} - {end}]")
        if speaker:
            lines.append(f"**Speaker**: {speaker}")
        lines.append("")
        lines.append(summary)
        lines.append("")

    return "\n".join(lines)


def format_segments_json(analysis: VideoAnalysis, segments: list[dict]) -> str:
    """Format segment analysis as JSON."""
    usage_dict = None
    if analysis.usage:
        u = analysis.usage
        usage_dict = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "total_tokens": u.total_tokens,
            "input_cost_usd": u.input_cost,
            "output_cost_usd": u.output_cost,
            "total_cost_usd": u.total_cost,
        }
    return json.dumps(
        {
            "url": analysis.url,
            "processed_at": analysis.processed_at,
            "model": analysis.model,
            "segments": segments,
            "usage": usage_dict,
            "error": analysis.error,
        },
        indent=2,
    )


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    # Replace spaces with underscores, remove non-alphanumeric chars except underscore/dash
    sanitized = re.sub(r"[^\w\s-]", "", name)
    sanitized = re.sub(r"\s+", "_", sanitized).strip("_")
    return sanitized[:80]  # Limit length


def split_video(
    file_path: str,
    segments: list[dict],
    output_dir: Path | None = None,
    verbose: bool = False,
) -> list[Path]:
    """Split a video file into segments using ffmpeg.

    Args:
        file_path: Path to the source video file.
        segments: List of segment dicts with start_time and end_time.
        output_dir: Directory for output files. Defaults to same directory as source.
        verbose: Print progress messages.

    Returns:
        List of paths to the created segment files.
    """
    if not shutil.which("ffmpeg"):
        raise click.ClickException(
            "ffmpeg is required for --split but was not found on PATH"
        )

    source = Path(file_path).resolve()
    dest_dir = output_dir or source.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []

    for seg in segments:
        num = seg.get("segment_number", 0)
        title = _sanitize_filename(seg.get("title", f"segment_{num}"))
        start = seg.get("start_time", "00:00:00")
        end = seg.get("end_time", "")

        out_name = f"{source.stem}_{num:02d}_{title}{source.suffix}"
        out_path = dest_dir / out_name

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-ss",
            start,
            "-to",
            end,
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(out_path),
        ]

        if verbose:
            click.echo(
                f"  Splitting segment {num}: {title} ({start} - {end})", err=True
            )

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            click.echo(
                f"  Warning: ffmpeg failed for segment {num}: {result.stderr[:200]}",
                err=True,
            )
        else:
            created_files.append(out_path)
            if verbose:
                size_mb = out_path.stat().st_size / (1024 * 1024)
                click.echo(f"    Created: {out_path.name} ({size_mb:.1f} MB)", err=True)

    return created_files


def format_output_markdown(analysis: VideoAnalysis) -> str:
    """Format analysis as markdown."""
    if analysis.error:
        return f"# Error Processing Video\n\n**URL**: {analysis.url}\n\n**Error**: {analysis.error}\n"

    usage_section = ""
    if analysis.usage:
        u = analysis.usage
        usage_section = f"""
**Usage**: {u.input_tokens:,} input + {u.output_tokens:,} output = {u.total_tokens:,} tokens
**Cost**: ${u.total_cost:.6f} (${u.input_cost:.6f} input + ${u.output_cost:.6f} output)
"""

    output = f"""# Video Analysis

**URL**: {analysis.url}
**Processed**: {analysis.processed_at}
**Model**: {analysis.model}{usage_section}

---

{analysis.raw_response}
"""
    return output


def format_output_json(analysis: VideoAnalysis) -> str:
    """Format analysis as JSON."""
    usage_dict = None
    if analysis.usage:
        u = analysis.usage
        usage_dict = {
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "total_tokens": u.total_tokens,
            "input_cost_usd": u.input_cost,
            "output_cost_usd": u.output_cost,
            "total_cost_usd": u.total_cost,
        }
    return json.dumps(
        {
            "url": analysis.url,
            "title": analysis.title,
            "processed_at": analysis.processed_at,
            "model": analysis.model,
            "content": analysis.raw_response,
            "summary": analysis.summary,
            "usage": usage_dict,
            "error": analysis.error,
        },
        indent=2,
    )


def get_safe_filename(input_source: str) -> str:
    """Generate safe filename from input source (URL, file path, or GCS URI)."""
    # Check if it's a local file
    path = Path(input_source)
    if path.exists() and path.is_file():
        return path.stem  # Return filename without extension

    # Check if it's a GCS URI
    if input_source.startswith("gs://"):
        filename = input_source.split("/")[-1]
        return Path(filename).stem

    # Check if it's a YouTube URL
    video_id = extract_video_id(input_source)
    if video_id:
        return f"video_{video_id}"

    return f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@click.command()
@click.argument("input", required=False)
@click.option(
    "--batch",
    "-b",
    type=click.Path(exists=True),
    help="File containing YouTube URLs or local file paths (one per line)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file or directory (for batch mode)",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="Output format (default: markdown)",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["comprehensive", "concise", "transcript", "segments"]),
    default="comprehensive",
    help="Analysis mode (default: comprehensive)",
)
@click.option(
    "--prompt",
    "-p",
    type=str,
    help="Custom prompt for analysis (overrides --mode)",
)
@click.option(
    "--model",
    type=click.Choice(
        [
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ]
    ),
    default="gemini-3-pro-preview",
    help="Gemini model to use (default: gemini-3-pro-preview)",
)
@click.option(
    "--api-key",
    envvar="GEMINI_API_KEY",
    help="Gemini API key (or set GEMINI_API_KEY env var)",
)
@click.option(
    "--vertex",
    is_flag=True,
    help="Use Vertex AI authentication (requires gcloud auth application-default login)",
)
@click.option(
    "--project",
    type=str,
    help="GCP project for Vertex AI (or set GOOGLE_CLOUD_PROJECT)",
)
@click.option(
    "--location",
    type=str,
    default=None,
    help="GCP location for Vertex AI (default: global for gemini-3-*, us-central1 for others)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output",
)
@click.option(
    "--split",
    is_flag=True,
    help="Split video into segments using ffmpeg (requires --mode segments and a local file)",
)
@click.version_option()
def main(
    input: str | None,
    batch: str | None,
    output: str | None,
    output_format: str,
    mode: str,
    prompt: str | None,
    model: str,
    api_key: str | None,
    vertex: bool,
    project: str | None,
    location: str | None,
    verbose: bool,
    split: bool,
):
    """
    Process videos using Google's Gemini API.

    Supports YouTube URLs and local video files.
    Extracts comprehensive content including transcripts and visual descriptions
    (slides, diagrams, charts, demonstrations).

    \b
    Examples:
        # Process a YouTube video
        yt-process "https://www.youtube.com/watch?v=VIDEO_ID"

        # Process a local video file
        yt-process ./video.mp4

        # Save to file
        yt-process ./presentation.mp4 -o analysis.md

        # Quick summary mode
        yt-process "https://youtube.com/watch?v=XYZ" -m concise

        # Transcript only
        yt-process ./meeting.mp4 -m transcript

        # Identify video segments
        yt-process ./meeting.mp4 -m segments

        # Identify segments and split into separate files
        yt-process ./meeting.mp4 -m segments --split

        # Batch process from file
        yt-process --batch inputs.txt -o ./output/

        # JSON output
        yt-process ./video.mp4 -f json

    \b
    Supported Video Formats:
        .mp4, .mpeg, .mov, .avi, .webm, .wmv, .flv, .mkv, .3gp

    \b
    Authentication:
        Option 1 - API Key:
            export GEMINI_API_KEY="your-key"
            yt-process "URL"

        Option 2 - Vertex AI (uses gcloud ADC):
            gcloud auth application-default login
            yt-process "URL" --vertex --project YOUR_PROJECT

    \b
    Environment Variables:
        GEMINI_API_KEY           Google Gemini API key
        GOOGLE_API_KEY           Alternative API key variable
        GOOGLE_CLOUD_PROJECT     GCP project for Vertex AI
        GOOGLE_CLOUD_LOCATION    GCP location (default: us-central1)
        GOOGLE_GENAI_USE_VERTEXAI  Set to "true" to auto-enable Vertex AI
    """
    if not input and not batch:
        raise click.ClickException(
            "Either INPUT (URL or file path) or --batch file is required"
        )

    if input and batch:
        raise click.ClickException("Cannot specify both INPUT and --batch")

    if split and mode != "segments" and not prompt:
        raise click.ClickException("--split requires --mode segments")

    # Auto-detect location based on model if not specified
    if location is None:
        if model.startswith("gemini-3"):
            location = "global"
        else:
            location = "us-central1"

    # Initialize client
    client = get_gemini_client(
        api_key=api_key,
        use_vertex=vertex,
        project=project,
        location=location,
    )

    # Determine prompt
    is_segments_mode = mode == "segments" and not prompt
    analysis_prompt = prompt if prompt else PROMPTS[mode]

    # Collect inputs to process (URLs or file paths)
    inputs: list[str] = []
    if batch:
        batch_path = Path(batch)
        inputs = [
            line.strip()
            for line in batch_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if verbose:
            click.echo(f"Found {len(inputs)} inputs in batch file")
    else:
        inputs = [input]  # type: ignore

    # Determine output handling
    is_batch = len(inputs) > 1
    output_dir_path: Path | None = None
    output_file: Path | None = None

    if output:
        output_path = Path(output)
        if is_batch or output_path.is_dir():
            output_dir_path = output_path
            output_dir_path.mkdir(parents=True, exist_ok=True)
        else:
            output_file = output_path

    # Process videos
    results: list[VideoAnalysis] = []

    formatter = (
        format_output_json if output_format == "json" else format_output_markdown
    )
    extension = "json" if output_format == "json" else "md"

    with click.progressbar(
        inputs,
        label="Processing videos",
        show_pos=True,
        item_show_func=lambda x: x[:50] + "..." if x and len(x) > 50 else x,
    ) as progress_inputs:
        for video_input in progress_inputs:
            if verbose:
                click.echo(f"\nProcessing: {video_input}")

            # Use response schema for segments mode to guarantee valid JSON
            schema = SEGMENTS_SCHEMA if is_segments_mode else None

            # For segments mode, detect video duration and inject into prompt
            video_prompt = analysis_prompt
            if is_segments_mode and is_local_file(video_input):
                duration = get_video_duration(video_input)
                if duration:
                    duration_line = (
                        f"\nThe video is exactly {duration} long. "
                        f"You MUST cover from 00:00:00 to {duration}.\n"
                    )
                    if verbose:
                        click.echo(f"  Detected video duration: {duration}", err=True)
                else:
                    duration_line = ""
                video_prompt = analysis_prompt.format(duration_line=duration_line)
            elif is_segments_mode:
                video_prompt = analysis_prompt.format(duration_line="")

            # Detect input type and process accordingly
            if is_local_file(video_input):
                analysis = process_local_file(
                    client, video_input, video_prompt, model, verbose, schema
                )
            elif is_gcs_uri(video_input):
                analysis = process_gcs_uri(
                    client, video_input, video_prompt, model, verbose, schema
                )
            else:
                analysis = process_video(
                    client, video_input, video_prompt, model, schema
                )
            results.append(analysis)

            # Segments mode: parse and format with segment-specific formatters
            if is_segments_mode and not analysis.error:
                segments = parse_segments(analysis.raw_response)

                if output_format == "json":
                    formatted = format_segments_json(analysis, segments)
                else:
                    formatted = format_segments_markdown(analysis, segments)

                # Split video if requested
                if split and is_local_file(video_input):
                    split_dir = output_dir_path or Path(video_input).resolve().parent
                    click.echo(
                        f"\nSplitting video into {len(segments)} segments...",
                        err=True,
                    )
                    created = split_video(video_input, segments, split_dir, verbose)
                    click.echo(
                        f"Created {len(created)} segment files in {split_dir}",
                        err=True,
                    )
                elif split and not is_local_file(video_input):
                    click.echo(
                        "Warning: --split only works with local video files",
                        err=True,
                    )
            else:
                formatted = formatter(analysis)

            if output_dir_path:
                # Batch mode: write each to separate file
                filename = f"{get_safe_filename(video_input)}.{extension}"
                file_path = output_dir_path / filename
                file_path.write_text(formatted)
                if verbose:
                    click.echo(f"  Saved to: {file_path}")
            elif output_file and not is_batch:
                # Single file output
                output_file.write_text(formatted)
                if verbose:
                    click.echo(f"Saved to: {output_file}")
            elif not output:
                # Print to stdout
                click.echo(formatted)

    # Summary for batch mode
    if is_batch:
        successful = sum(1 for r in results if not r.error)
        failed = sum(1 for r in results if r.error)
        click.echo(
            f"\nProcessed {len(results)} videos: {successful} successful, {failed} failed"
        )

        if failed > 0 and verbose:
            click.echo("\nFailed videos:")
            for r in results:
                if r.error:
                    click.echo(f"  - {r.url}: {r.error}")


if __name__ == "__main__":
    main()
