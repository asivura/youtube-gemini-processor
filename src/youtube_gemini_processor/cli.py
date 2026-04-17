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
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import click


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
    summary: str = ""
    raw_response: str = ""
    processed_at: str = ""
    model: str = ""
    error: str | None = None
    usage: UsageStats | None = None


# Default comprehensive analysis prompt - optimized for maximum detail
DEFAULT_PROMPT = """You are an expert content analyst. Analyze this media with extreme thoroughness and produce a comprehensive markdown document.
{duration_line}
# INSTRUCTIONS
- Watch/listen to the ENTIRE media carefully
- Capture EVERY piece of information: spoken words, visuals, text on screen
- Be extremely detailed - this document should allow someone to fully understand the media without watching or listening to it
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


CONCISE_PROMPT = """Analyze this media. Provide:
{duration_line}
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


TRANSCRIPT_ONLY_PROMPT = """Provide a complete transcript of this media.
{duration_line}
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

# Pricing per 1M tokens
# https://ai.google.dev/pricing
MODEL_PRICING = {
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},  # Per 1M tokens
    "gemini-3-pro-preview": {"input": 1.25, "output": 10.00},
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}

# Maximum output tokens per model
MODEL_MAX_OUTPUT_TOKENS = {
    "gemini-3-flash-preview": 65536,
    "gemini-3-pro-preview": 65536,
    "gemini-3.1-pro-preview": 65536,
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
            or os.environ.get("YT_PROCESS_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("GCP_PROJECT")
            or os.environ.get("CLOUDSDK_CORE_PROJECT")
        )
        gcp_location = (
            location
            or os.environ.get("YT_PROCESS_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION")
            or os.environ.get("CLOUDSDK_COMPUTE_REGION")
            or "global"
        )

        if not gcp_project:
            raise click.ClickException(
                "Vertex AI requires a GCP project. Set YT_PROCESS_PROJECT "
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


def _normalize_timestamp_to_hhmmss(timestamp: str) -> str:
    """Normalize a MM:SS or HH:MM:SS timestamp to HH:MM:SS format."""
    parts = timestamp.split(":")
    if len(parts) == 2:
        return f"00:{parts[0].zfill(2)}:{parts[1]}"
    if len(parts) == 3:
        return f"{parts[0].zfill(2)}:{parts[1]}:{parts[2]}"
    return timestamp


def fetch_youtube_chapters(url: str) -> list[dict]:
    """Fetch chapter timestamps from a YouTube video description.

    Scrapes the YouTube page and extracts chapter markers from the description.
    Chapters are timestamps in the format "(HH:MM:SS) Title" or "(MM:SS) Title".

    Returns:
        List of segment dicts with segment_number, start_time, end_time, title.
        Empty list if no chapters found.
    """
    video_id = extract_video_id(url)
    if not video_id:
        return []

    fetch_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        req = urllib.request.Request(
            fetch_url,
            headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    # Extract description from page data
    description = ""
    for pattern in [
        r'"shortDescription":"(.*?)"',
        r'"description":\{"simpleText":"(.*?)"\}',
    ]:
        match = re.search(pattern, html)
        if match:
            description = match.group(1).encode().decode("unicode_escape")
            break

    if not description:
        return []

    # Parse chapter timestamps from description
    # Matches patterns like: (00:00) Title, 0:00 Title, 00:00:00 Title
    chapter_pattern = re.compile(
        r"^\(?(\d{1,2}:\d{2}(?::\d{2})?)\)?\s+(.+)$", re.MULTILINE
    )
    matches = list(chapter_pattern.finditer(description))

    if len(matches) < 2:
        return []

    chapters = []
    for i, m in enumerate(matches):
        start_time = m.group(1)
        title = m.group(2).strip()

        start_time = _normalize_timestamp_to_hhmmss(start_time)

        # End time is start of next chapter, or empty for last
        if i + 1 < len(matches):
            end_time = _normalize_timestamp_to_hhmmss(matches[i + 1].group(1))
        else:
            end_time = ""

        chapters.append(
            {
                "segment_number": i + 1,
                "start_time": start_time,
                "end_time": end_time,
                "title": title,
                "speaker": "",
                "summary": "",
            }
        )

    return chapters


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

# Supported audio MIME types
AUDIO_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".aac": "audio/aac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
}

MediaKind = Literal["video", "audio"]


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


def parse_timestamp_to_seconds(timestamp: str) -> str:
    """Parse a timestamp string to seconds format for the API.

    Accepts formats:
        - "123s" or "123" (raw seconds)
        - "MM:SS" (e.g., "5:30")
        - "HH:MM:SS" (e.g., "1:05:30")

    Returns:
        String in "{seconds}s" format (e.g., "330s").
    """
    timestamp = timestamp.strip()

    # Already in seconds format
    if timestamp.endswith("s"):
        return timestamp
    if timestamp.isdigit():
        return f"{timestamp}s"

    parts = timestamp.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = int(parts[0]), int(parts[1])
            return f"{minutes * 60 + seconds}s"
        if len(parts) == 3:
            hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{hours * 3600 + minutes * 60 + seconds}s"
    except ValueError:
        pass

    raise click.ClickException(
        f"Invalid timestamp format: {timestamp}\n"
        "Expected: SS, MM:SS, or HH:MM:SS (e.g., 90, 1:30, 0:01:30)"
    )


def parse_clip_range(clip: str) -> tuple[str, str]:
    """Parse a clip range string into start and end offsets.

    Accepts format: "START-END" where START and END are timestamps.
    Examples: "1:30-5:00", "0:01:30-0:05:00", "90-300", "90s-300s"

    Returns:
        Tuple of (start_offset, end_offset) in "{seconds}s" format.
    """
    if "-" not in clip:
        raise click.ClickException(
            f"Invalid clip format: {clip}\n"
            "Expected: START-END (e.g., 1:30-5:00 or 90-300)"
        )

    # Split on last hyphen to handle negative edge cases
    # But timestamps don't have negatives, so split on first hyphen
    # that separates two timestamp parts
    # Handle HH:MM:SS-HH:MM:SS by finding the separator hyphen
    # A separator hyphen is one that's NOT preceded by a colon
    parts = clip.split("-")
    if len(parts) == 2:
        start_str, end_str = parts
    elif len(parts) > 2:
        # Could be HH:MM:SS-HH:MM:SS which has no extra hyphens,
        # or raw seconds like 90-300. Try first hyphen after pos 0.
        dash_idx = clip.find("-", 1)
        start_str = clip[:dash_idx]
        end_str = clip[dash_idx + 1 :]
    else:
        raise click.ClickException(
            f"Invalid clip format: {clip}\n"
            "Expected: START-END (e.g., 1:30-5:00 or 90-300)"
        )

    return parse_timestamp_to_seconds(start_str), parse_timestamp_to_seconds(end_str)


MEDIA_RESOLUTION_MAP = {
    "low": "MEDIA_RESOLUTION_LOW",
    "medium": "MEDIA_RESOLUTION_MEDIUM",
    "high": "MEDIA_RESOLUTION_HIGH",
}


def build_media_part(
    file_uri: str,
    mime_type: str,
    *,
    kind: MediaKind = "video",
    fps: float | None = None,
    clip_start: str | None = None,
    clip_end: str | None = None,
):
    """Build a media Part with optional VideoMetadata.

    Args:
        file_uri: The file URI (Files API, GCS, or YouTube URL).
        mime_type: MIME type of the media.
        kind: Media kind, "video" or "audio". Audio never attaches VideoMetadata.
        fps: Custom frames per second for sampling (video only).
        clip_start: Start offset in "{seconds}s" format.
        clip_end: End offset in "{seconds}s" format.
    """
    from google.genai import types

    part_kwargs: dict = {
        "file_data": types.FileData(file_uri=file_uri, mime_type=mime_type),
    }

    if kind == "video":
        vm_kwargs: dict = {}
        if fps is not None:
            vm_kwargs["fps"] = fps
        if clip_start is not None:
            vm_kwargs["start_offset"] = clip_start
        if clip_end is not None:
            vm_kwargs["end_offset"] = clip_end

        if vm_kwargs:
            part_kwargs["video_metadata"] = types.VideoMetadata(**vm_kwargs)
    elif clip_start is not None or clip_end is not None:
        vm_kwargs = {}
        if clip_start is not None:
            vm_kwargs["start_offset"] = clip_start
        if clip_end is not None:
            vm_kwargs["end_offset"] = clip_end
        part_kwargs["video_metadata"] = types.VideoMetadata(**vm_kwargs)

    return types.Part(**part_kwargs)


def build_generate_config(
    model: str,
    *,
    response_schema: dict | None = None,
    media_resolution: str | None = None,
):
    """Build a GenerateContentConfig with optional media resolution.

    Args:
        model: Model name (used to determine max output tokens).
        response_schema: Optional JSON schema for structured output.
        media_resolution: One of "low", "medium", "high", or None.
    """
    from google.genai import types

    config_kwargs: dict = {
        "max_output_tokens": get_max_output_tokens(model),
    }
    if response_schema:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema
    if media_resolution:
        config_kwargs["media_resolution"] = media_resolution

    return types.GenerateContentConfig(**config_kwargs)


def is_gcs_uri(uri: str) -> bool:
    """Check if input is a Google Cloud Storage URI."""
    return uri.startswith("gs://")


def is_audio_input(input_str: str) -> bool:
    """Return True if the input points to an audio file by its extension."""
    if is_youtube_url(input_str) or is_files_api_ref(input_str):
        return False
    ext = Path(input_str).suffix.lower()
    return ext in AUDIO_MIME_TYPES


def is_files_api_ref(input_str: str) -> bool:
    """Check if input is a Files API reference.

    Accepts either the short name format (files/abc123) or the full URI
    (https://generativelanguage.googleapis.com/v1beta/files/abc123).
    """
    return input_str.startswith("files/") or (
        "generativelanguage.googleapis.com" in input_str and "/files/" in input_str
    )


def normalize_files_api_ref(input_str: str) -> str:
    """Extract the Files API name (files/xxx) from a name or full URI.

    Args:
        input_str: Either "files/abc123" or a full URI containing "/files/abc123".

    Returns:
        The normalized name in "files/xxx" format.

    Raises:
        click.ClickException: If the input cannot be parsed as a Files API reference.
    """
    if input_str.startswith("files/"):
        return input_str
    match = re.search(r"(files/[a-zA-Z0-9_-]+)", input_str)
    if match:
        return match.group(1)
    raise click.ClickException(
        f"Invalid Files API reference: {input_str}\n"
        "Expected format: files/abc123 or full URI"
    )


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
            timeout=300,
        )
        if result.returncode != 0:
            return None
        seconds = float(result.stdout.strip())
        return _format_duration(seconds)
    except (ValueError, OSError):
        return None


def get_video_duration_gcs(gcs_uri: str) -> str | None:
    """Get video duration from a GCS URI using ffprobe with authenticated HTTPS.

    Converts gs://bucket/path to an HTTPS URL and uses an OAuth2 access token
    for authentication. ffprobe only reads the file header, not the full file.

    Returns None if ffprobe is not available, auth fails, or probing fails.
    """
    if not shutil.which("ffprobe"):
        return None

    match = re.match(r"gs://([^/]+)/(.+)", gcs_uri)
    if not match:
        return None

    bucket, path = match.groups()
    https_url = (
        f"https://storage.googleapis.com/{bucket}/{urllib.parse.quote(path, safe='/')}"
    )

    try:
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default()
        credentials.refresh(google.auth.transport.requests.Request())
        token = credentials.token
    except Exception:
        return None

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-headers",
                f"Authorization: Bearer {token}\r\n",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                https_url,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            return None
        seconds = float(result.stdout.strip())
        return _format_duration(seconds)
    except (ValueError, OSError):
        return None


def get_media_mime_type(file_path: Path) -> tuple[str, MediaKind]:
    """Get MIME type and media kind for a media file based on extension.

    Returns:
        Tuple of (mime_type, kind) where kind is "video" or "audio".
    """
    ext = file_path.suffix.lower()
    video_mime = VIDEO_MIME_TYPES.get(ext)
    if video_mime:
        return video_mime, "video"
    audio_mime = AUDIO_MIME_TYPES.get(ext)
    if audio_mime:
        return audio_mime, "audio"
    raise click.ClickException(
        f"Unsupported media format: {ext}\n"
        f"Supported video formats: {', '.join(VIDEO_MIME_TYPES.keys())}\n"
        f"Supported audio formats: {', '.join(AUDIO_MIME_TYPES.keys())}"
    )


def mime_type_for_extension(ext: str, default_kind: MediaKind = "video") -> str:
    """Look up a MIME type by extension, falling back to a default for the given kind."""
    ext = ext.lower()
    if ext in VIDEO_MIME_TYPES:
        return VIDEO_MIME_TYPES[ext]
    if ext in AUDIO_MIME_TYPES:
        return AUDIO_MIME_TYPES[ext]
    return "video/mp4" if default_kind == "video" else "audio/mpeg"


def _call_gemini_and_parse(
    client,
    media_part,
    model: str,
    prompt: str,
    analysis: VideoAnalysis,
    fallback_title: str = "",
    response_schema: dict | None = None,
    media_resolution: str | None = None,
) -> None:
    """Call Gemini API and parse the response into a VideoAnalysis.

    Handles the generate_content call, usage stats extraction,
    title extraction, and summary extraction. Mutates `analysis` in place.

    Args:
        client: Gemini API client.
        media_part: The media Part object (video or audio) for the request.
        model: Model name.
        prompt: The analysis prompt text.
        analysis: VideoAnalysis object to populate.
        fallback_title: Title to use if none is extracted from the response.
        response_schema: Optional JSON schema for structured output.
        media_resolution: Optional media resolution setting.
    """
    from google.genai import types

    config = build_generate_config(
        model,
        response_schema=response_schema,
        media_resolution=media_resolution,
    )

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[media_part, types.Part(text=prompt)],
            )
        ],
        config=config,
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
    elif fallback_title:
        analysis.title = fallback_title

    # Extract summary if present
    summary_match = re.search(
        r"(?:## (?:5\. )?summary|summary:)\s*\n(.*?)(?=\n## |\n\*\*|\Z)",
        response.text,
        re.IGNORECASE | re.DOTALL,
    )
    if summary_match:
        analysis.summary = summary_match.group(1).strip()


def process_local_file(
    client,
    file_path: str,
    prompt: str,
    model: str = "gemini-3-pro-preview",
    verbose: bool = False,
    response_schema: dict | None = None,
    fps: float | None = None,
    clip_start: str | None = None,
    clip_end: str | None = None,
    media_resolution: str | None = None,
) -> VideoAnalysis:
    """Process a local media file with Gemini API using Files API."""
    path = Path(file_path).resolve()

    analysis = VideoAnalysis(
        url=str(path),
        processed_at=datetime.now().isoformat(),
        model=model,
    )

    try:
        mime_type, kind = get_media_mime_type(path)
        file_size_mb = path.stat().st_size / (1024 * 1024)

        if verbose:
            click.echo(f"  Uploading {path.name} ({file_size_mb:.1f} MB)...", err=True)

        # Upload file using Files API
        uploaded_file = client.files.upload(file=str(path))

        click.echo(
            f"  Uploaded: {uploaded_file.name} (reuse with: yt-process {uploaded_file.name})",
            err=True,
        )

        # Wait for file to be processed
        while uploaded_file.state.name == "PROCESSING":
            if verbose:
                click.echo("  Waiting for file processing...", err=True)
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name == "FAILED":
            raise click.ClickException(f"File processing failed: {uploaded_file.name}")

        media_part = build_media_part(
            uploaded_file.uri,
            mime_type,
            kind=kind,
            fps=fps,
            clip_start=clip_start,
            clip_end=clip_end,
        )
        _call_gemini_and_parse(
            client,
            media_part,
            model,
            prompt,
            analysis,
            fallback_title=path.stem,
            response_schema=response_schema,
            media_resolution=media_resolution,
        )

    except Exception as e:
        analysis.error = str(e)

    return analysis


def process_files_api_ref(
    client,
    file_ref: str,
    prompt: str,
    model: str = "gemini-3-pro-preview",
    verbose: bool = False,
    response_schema: dict | None = None,
    fps: float | None = None,
    clip_start: str | None = None,
    clip_end: str | None = None,
    media_resolution: str | None = None,
) -> VideoAnalysis:
    """Process a video using an existing Files API reference.

    Skips upload entirely and uses a previously uploaded file.
    Files API references expire after 48 hours.
    """

    file_name = normalize_files_api_ref(file_ref)

    analysis = VideoAnalysis(
        url=file_ref,
        processed_at=datetime.now().isoformat(),
        model=model,
    )

    try:
        if verbose:
            click.echo(f"  Looking up file: {file_name}", err=True)

        try:
            file_info = client.files.get(name=file_name)
        except Exception as e:
            raise click.ClickException(
                f"File not found: {file_name}\n"
                "It may have expired (48h limit) or been deleted.\n"
                f"Error: {e}"
            ) from None

        # Wait if still processing
        while file_info.state.name == "PROCESSING":
            if verbose:
                click.echo("  Waiting for file processing...", err=True)
            time.sleep(2)
            file_info = client.files.get(name=file_name)

        if file_info.state.name == "FAILED":
            raise click.ClickException(f"File processing failed: {file_name}")

        display = getattr(file_info, "display_name", None) or file_name
        ext = Path(display).suffix.lower()
        # Use MIME type from file metadata, fall back based on file extension
        fallback_mime = mime_type_for_extension(ext)
        mime_type = getattr(file_info, "mime_type", None) or fallback_mime
        kind: MediaKind = "audio" if mime_type.startswith("audio/") else "video"

        if verbose:
            click.echo(f"  Using file: {file_info.name} ({mime_type})", err=True)

        media_part = build_media_part(
            file_info.uri,
            mime_type,
            kind=kind,
            fps=fps,
            clip_start=clip_start,
            clip_end=clip_end,
        )
        _call_gemini_and_parse(
            client,
            media_part,
            model,
            prompt,
            analysis,
            fallback_title=Path(display).stem,
            response_schema=response_schema,
            media_resolution=media_resolution,
        )

    except click.ClickException:
        raise
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
    fps: float | None = None,
    clip_start: str | None = None,
    clip_end: str | None = None,
    media_resolution: str | None = None,
) -> VideoAnalysis:
    """Process a media file from Google Cloud Storage with Gemini API."""
    filename = gcs_uri.split("/")[-1]
    ext = Path(filename).suffix.lower()

    if ext in AUDIO_MIME_TYPES:
        mime_type = AUDIO_MIME_TYPES[ext]
        kind: MediaKind = "audio"
    else:
        mime_type = VIDEO_MIME_TYPES.get(ext, "video/mp4")
        kind = "video"

    analysis = VideoAnalysis(
        url=gcs_uri,
        processed_at=datetime.now().isoformat(),
        model=model,
    )

    try:
        if verbose:
            click.echo(f"  Processing GCS file: {filename}", err=True)

        media_part = build_media_part(
            gcs_uri,
            mime_type,
            kind=kind,
            fps=fps,
            clip_start=clip_start,
            clip_end=clip_end,
        )
        _call_gemini_and_parse(
            client,
            media_part,
            model,
            prompt,
            analysis,
            fallback_title=Path(filename).stem,
            response_schema=response_schema,
            media_resolution=media_resolution,
        )

    except Exception as e:
        analysis.error = str(e)

    return analysis


def process_video(
    client,
    url: str,
    prompt: str,
    model: str = "gemini-3-pro-preview",
    response_schema: dict | None = None,
    fps: float | None = None,
    clip_start: str | None = None,
    clip_end: str | None = None,
    media_resolution: str | None = None,
) -> VideoAnalysis:
    """Process a single YouTube video with Gemini API."""
    normalized_url = validate_youtube_url(url)

    analysis = VideoAnalysis(
        url=normalized_url,
        processed_at=datetime.now().isoformat(),
        model=model,
    )

    try:
        media_part = build_media_part(
            normalized_url,
            "video/mp4",
            kind="video",
            fps=fps,
            clip_start=clip_start,
            clip_end=clip_end,
        )
        _call_gemini_and_parse(
            client,
            media_part,
            model,
            prompt,
            analysis,
            response_schema=response_schema,
            media_resolution=media_resolution,
        )

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
            timeout=300,
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


def _process_single_chapter(
    client,
    url: str,
    chapter: dict,
    total_chapters: int,
    analysis_prompt: str,
    model: str,
    output_dir: Path,
    formatter,
    extension: str,
    verbose: bool,
    fps: float | None,
    media_resolution: str | None,
) -> tuple[Path | None, UsageStats | None, str]:
    """Process a single chapter clip. Returns (output_path, usage, status_msg)."""
    num = chapter.get("segment_number", 0)
    title = chapter.get("title", "untitled")
    start = chapter.get("start_time", "")
    end = chapter.get("end_time", "")

    if not start:
        return None, None, f"  [{num}/{total_chapters}] Skipped: no start time"

    try:
        clip_start = parse_timestamp_to_seconds(start)
        clip_end = parse_timestamp_to_seconds(end) if end else None
    except click.ClickException:
        return None, None, f"  [{num}/{total_chapters}] Skipped: invalid timestamps"

    safe_title = _sanitize_filename(title)
    filename = f"{num:02d}_{safe_title}.{extension}"
    out_path = output_dir / filename

    try:
        analysis = process_video(
            client,
            url,
            analysis_prompt,
            model,
            response_schema=None,
            fps=fps,
            clip_start=clip_start,
            clip_end=clip_end,
            media_resolution=media_resolution,
        )
    except Exception as e:
        return None, None, f"  [{num}/{total_chapters}] {title}: Error - {e}"

    if analysis.error:
        return (
            None,
            None,
            f"  [{num}/{total_chapters}] {title}: Error - {analysis.error}",
        )

    formatted = formatter(analysis)
    out_path.write_text(formatted)

    cost_str = f" (${analysis.usage.total_cost:.4f})" if analysis.usage else ""
    msg = f"  [{num}/{total_chapters}] {title} ({start} - {end or 'end'}){cost_str}"
    return out_path, analysis.usage, msg


def split_youtube_video(
    client,
    url: str,
    chapters: list[dict],
    analysis_prompt: str,
    model: str,
    output_dir: Path,
    output_format: str,
    verbose: bool = False,
    fps: float | None = None,
    media_resolution: str | None = None,
    max_workers: int = 4,
) -> list[Path]:
    """Process a YouTube video in chunks, one per chapter (parallelized).

    Each chapter is processed as an independent --clip call to the Gemini API,
    producing a separate output file per chapter. Chapters are processed
    concurrently using a thread pool.

    Args:
        client: Gemini API client.
        url: YouTube URL.
        chapters: List of chapter dicts with start_time, end_time, title.
        analysis_prompt: The prompt to use for each chunk.
        model: Model name.
        output_dir: Directory for output files.
        output_format: "markdown" or "json".
        verbose: Print progress messages.
        fps: Frame sampling rate.
        media_resolution: Video resolution setting.
        max_workers: Maximum concurrent API calls (default: 4).

    Returns:
        List of paths to the created output files.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    output_dir.mkdir(parents=True, exist_ok=True)
    formatter = (
        format_output_json if output_format == "json" else format_output_markdown
    )
    extension = "json" if output_format == "json" else "md"
    created_files = []
    total_usage = UsageStats()
    total = len(chapters)

    click.echo(f"Processing {total} chapters with {max_workers} workers...", err=True)

    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chapter in chapters:
            future = executor.submit(
                _process_single_chapter,
                client,
                url,
                chapter,
                total,
                analysis_prompt,
                model,
                output_dir,
                formatter,
                extension,
                verbose,
                fps,
                media_resolution,
            )
            futures[future] = chapter

        for future in as_completed(futures):
            out_path, usage, msg = future.result()
            click.echo(msg, err=True)

            if out_path:
                created_files.append(out_path)
            if usage:
                total_usage.input_tokens += usage.input_tokens
                total_usage.output_tokens += usage.output_tokens
                total_usage.total_tokens += usage.total_tokens
                total_usage.input_cost += usage.input_cost
                total_usage.output_cost += usage.output_cost
                total_usage.total_cost += usage.total_cost

    # Sort by filename for consistent output
    created_files.sort()

    click.echo(
        f"\nProcessed {len(created_files)}/{total} chapters. "
        f"Total: {total_usage.total_tokens:,} tokens, ${total_usage.total_cost:.4f}",
        err=True,
    )

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

    # Check if it's a Files API reference
    if is_files_api_ref(input_source):
        name = normalize_files_api_ref(input_source)
        return name.replace("/", "_")

    # Check if it's a GCS URI
    if input_source.startswith("gs://"):
        filename = input_source.split("/")[-1]
        return Path(filename).stem

    # Check if it's a YouTube URL
    video_id = extract_video_id(input_source)
    if video_id:
        return f"video_{video_id}"

    return f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _handle_file_management(
    client,
    list_files: bool,
    delete_file: str | None,
) -> None:
    """Handle --list-files and --delete-file operations."""
    if list_files:
        click.echo("Uploaded files:", err=True)
        found = False
        for f in client.files.list():
            found = True
            state = getattr(f.state, "name", "UNKNOWN")
            display = getattr(f, "display_name", "") or ""
            expire = getattr(f, "expiration_time", "") or ""
            line = f"  {f.name:<30} {state:<10}"
            if display:
                line += f" {display}"
            if expire:
                line += f"  (expires: {expire})"
            click.echo(line)
        if not found:
            click.echo("  (no files uploaded)")
    if delete_file:
        name = normalize_files_api_ref(delete_file)
        client.files.delete(name=name)
        click.echo(f"Deleted: {name}", err=True)


def _handle_upload_only(
    client,
    source: str,
    verbose: bool,
) -> None:
    """Handle --upload-only mode: upload file and print reference."""
    import time

    if not is_local_file(source):
        raise click.ClickException("--upload-only requires a local file path as input")

    path = Path(source).resolve()
    mime_type, _kind = get_media_mime_type(path)
    file_size_mb = path.stat().st_size / (1024 * 1024)

    click.echo(f"Uploading {path.name} ({file_size_mb:.1f} MB)...", err=True)
    uploaded_file = client.files.upload(file=str(path))
    click.echo("Waiting for processing...", err=True)

    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = client.files.get(name=uploaded_file.name)

    if uploaded_file.state.name == "FAILED":
        raise click.ClickException(f"File processing failed: {uploaded_file.name}")

    click.echo(uploaded_file.name)
    if verbose:
        click.echo(f"  URI: {uploaded_file.uri}", err=True)
        click.echo(f"  MIME type: {mime_type}", err=True)
        expire = getattr(uploaded_file, "expiration_time", None)
        if expire:
            click.echo(f"  Expires: {expire}", err=True)


def _handle_chapter_splitting(
    client,
    url: str,
    analysis_prompt: str,
    model: str,
    output: str | None,
    output_format: str,
    verbose: bool,
    fps: float | None,
    media_resolution: str | None,
    workers: int,
) -> None:
    """Handle YouTube chapter-based splitting (--split without segments mode)."""
    click.echo("Fetching YouTube chapters...", err=True)
    chapters = fetch_youtube_chapters(url)

    if not chapters:
        raise click.ClickException(
            "No YouTube chapters found in video description. "
            "Use --mode segments to detect chapters with AI, or process manually with --clip."
        )

    click.echo(f"Found {len(chapters)} chapters:", err=True)
    for ch in chapters:
        click.echo(
            f"  {ch['segment_number']:2d}. [{ch['start_time']} - {ch['end_time'] or 'end'}] "
            f"{ch['title']}",
            err=True,
        )

    # Determine output directory
    if output:
        split_out = Path(output)
    else:
        video_id = extract_video_id(url) or "youtube"
        split_out = Path(f"split_{video_id}")

    created = split_youtube_video(
        client=client,
        url=url,
        chapters=chapters,
        analysis_prompt=analysis_prompt,
        model=model,
        output_dir=split_out,
        output_format=output_format,
        verbose=verbose,
        fps=fps,
        media_resolution=media_resolution,
        max_workers=workers,
    )

    click.echo(f"\nCreated {len(created)} chapter files in {split_out}", err=True)


def _handle_output(
    formatted: str,
    video_input: str,
    extension: str,
    output_dir_path: Path | None,
    output_file: Path | None,
    is_batch: bool,
    verbose: bool,
) -> None:
    """Handle writing formatted output to file or stdout."""
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
    elif not output_dir_path and not output_file:
        # Print to stdout
        click.echo(formatted)


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
    type=click.Choice(list(MODEL_PRICING.keys())),
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
    help="GCP project for Vertex AI (or set YT_PROCESS_PROJECT)",
)
@click.option(
    "--location",
    type=str,
    default=None,
    help="GCP location for Vertex AI (default: global)",
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
    help="Split video into segments. Local files: uses ffmpeg (requires --mode segments). "
    "YouTube URLs: fetches chapters and processes each chunk independently.",
)
@click.option(
    "--workers",
    type=int,
    default=4,
    help="Number of parallel workers for --split with YouTube URLs (default: 4).",
)
@click.option(
    "--upload-only",
    is_flag=True,
    help="Upload file to Files API and print reference without processing",
)
@click.option(
    "--list-files",
    is_flag=True,
    help="List all files uploaded to the Files API",
)
@click.option(
    "--delete-file",
    type=str,
    default=None,
    help="Delete a Files API reference (e.g. files/abc123)",
)
@click.option(
    "--fps",
    type=float,
    default=None,
    help="Custom frame sampling rate (frames per second). Default: Gemini uses 1 FPS. "
    "Higher values capture more detail but increase token usage.",
)
@click.option(
    "--clip",
    type=str,
    default=None,
    help="Process only a clip of the video. Format: START-END "
    "(e.g., 1:30-5:00, 0:01:30-0:05:00, 90-300).",
)
@click.option(
    "--media-resolution",
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    default=None,
    help="Video resolution for processing. 'low' uses ~66 tokens/frame, "
    "'high' uses ~258 tokens/frame (default). Lower saves tokens for long videos.",
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
    workers: int,
    upload_only: bool,
    list_files: bool,
    delete_file: str | None,
    fps: float | None,
    clip: str | None,
    media_resolution: str | None,
):
    """
    Process videos using Google's Gemini API.

    Supports YouTube URLs, local video files, and Files API references.
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
    Files API (upload once, reuse for 48 hours):
        # Upload and get reference
        yt-process ./video.mp4 --upload-only

        # Process using saved reference (no re-upload)
        yt-process files/abc123 -m comprehensive
        yt-process files/abc123 -m segments

        # List uploaded files
        yt-process --list-files

        # Delete a file
        yt-process --delete-file files/abc123

    \b
    Video Processing Options:
        # Sample at 2 FPS (more detail, more tokens)
        yt-process ./video.mp4 --fps 2

        # Process only a clip (1:30 to 5:00)
        yt-process ./video.mp4 --clip 1:30-5:00

        # Use low resolution (saves tokens for long videos)
        yt-process ./video.mp4 --media-resolution low

        # Combine options
        yt-process files/abc123 --clip 0:00-10:00 --fps 0.5 --media-resolution low

    \b
    Supported Media Formats:
        Video: .mp4, .mpeg, .mov, .avi, .webm, .wmv, .flv, .mkv, .3gp
        Audio: .mp3, .m4a, .wav, .flac, .ogg, .aac, .aiff, .aif
        Note: --fps and --media-resolution apply to video only.

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
        YT_PROCESS_PROJECT       GCP project for Vertex AI (tool-specific)
        GOOGLE_CLOUD_PROJECT     GCP project for Vertex AI (fallback)
        YT_PROCESS_LOCATION      GCP location (tool-specific, preferred)
        GOOGLE_CLOUD_LOCATION    GCP location (fallback, default: global)
        GOOGLE_GENAI_USE_VERTEXAI  Set to "true" to auto-enable Vertex AI
    """
    # Auto-detect location based on model if not specified
    if location is None:
        # Default to global for all models to avoid regional quota limits
        location = "global"

    # Parse video processing options
    clip_start: str | None = None
    clip_end: str | None = None
    if clip:
        clip_start, clip_end = parse_clip_range(clip)

    resolved_media_resolution: str | None = None
    if media_resolution:
        resolved_media_resolution = MEDIA_RESOLUTION_MAP[media_resolution]

    # Handle file management operations (no INPUT required)
    if list_files or delete_file:
        client = get_gemini_client(
            api_key=api_key,
            use_vertex=vertex,
            project=project,
            location=location,
        )
        _handle_file_management(client, list_files, delete_file)
        return

    if not input and not batch:
        raise click.ClickException(
            "Either INPUT (URL or file path) or --batch file is required"
        )

    if input and batch:
        raise click.ClickException("Cannot specify both INPUT and --batch")

    if (
        split
        and input
        and not is_youtube_url(input)
        and mode != "segments"
        and not prompt
    ):
        raise click.ClickException("--split requires --mode segments for local files")

    if upload_only and batch:
        raise click.ClickException("--upload-only cannot be used with --batch")

    if input and is_audio_input(input):
        if fps is not None:
            raise click.ClickException("--fps is not supported for audio inputs")
        if media_resolution is not None:
            raise click.ClickException(
                "--media-resolution is not supported for audio inputs"
            )

    # Initialize client
    client = get_gemini_client(
        api_key=api_key,
        use_vertex=vertex,
        project=project,
        location=location,
    )

    # Handle upload-only mode
    if upload_only:
        if not input:
            raise click.ClickException(
                "--upload-only requires a local file path as input"
            )
        _handle_upload_only(client, input, verbose)
        return

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

    # YouTube chunked processing: --split with YouTube URL (not segments mode)
    if split and input and is_youtube_url(input) and not is_segments_mode:
        _handle_chapter_splitting(
            client=client,
            url=input,
            analysis_prompt=analysis_prompt,
            model=model,
            output=output,
            output_format=output_format,
            verbose=verbose,
            fps=fps,
            media_resolution=resolved_media_resolution,
            workers=workers,
        )
        return

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

            # Detect media duration and inject into prompt. Applied for all
            # built-in modes (comprehensive/concise/transcript/segments) but
            # skipped for custom --prompt since user text may contain literal
            # braces that would break str.format().
            video_prompt = analysis_prompt
            if not prompt:
                duration = None
                if is_local_file(video_input):
                    duration = get_video_duration(video_input)
                elif is_gcs_uri(video_input):
                    duration = get_video_duration_gcs(video_input)

                if duration:
                    duration_line = (
                        f"\nThe media is exactly {duration} long. "
                        f"You MUST cover from 00:00:00 to {duration}.\n"
                    )
                    if verbose:
                        click.echo(f"  Detected media duration: {duration}", err=True)
                else:
                    duration_line = ""
                video_prompt = analysis_prompt.format(duration_line=duration_line)

            # Detect input type and process accordingly
            # Common video processing kwargs
            video_kwargs = {
                "fps": fps,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "media_resolution": resolved_media_resolution,
            }

            if is_local_file(video_input):
                analysis = process_local_file(
                    client,
                    video_input,
                    video_prompt,
                    model,
                    verbose,
                    schema,
                    **video_kwargs,
                )
            elif is_files_api_ref(video_input):
                analysis = process_files_api_ref(
                    client,
                    video_input,
                    video_prompt,
                    model,
                    verbose,
                    schema,
                    **video_kwargs,
                )
            elif is_gcs_uri(video_input):
                analysis = process_gcs_uri(
                    client,
                    video_input,
                    video_prompt,
                    model,
                    verbose,
                    schema,
                    **video_kwargs,
                )
            else:
                analysis = process_video(
                    client,
                    video_input,
                    video_prompt,
                    model,
                    schema,
                    **video_kwargs,
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
                    # For non-local files, --split in segments mode just warns
                    click.echo(
                        "Note: --split with segments mode on non-local files only "
                        "outputs segment data. Use --split without --mode segments "
                        "on YouTube URLs for chunked processing.",
                        err=True,
                    )
            else:
                formatted = formatter(analysis)

            _handle_output(
                formatted,
                video_input,
                extension,
                output_dir_path,
                output_file,
                is_batch,
                verbose,
            )

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
