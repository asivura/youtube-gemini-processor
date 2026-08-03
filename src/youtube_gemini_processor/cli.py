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

import base64
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import click
import httpx


@dataclass
class UsageStats:
    """Token usage and cost statistics.

    `output_tokens` is the *billed* output count: Gemini charges reasoning
    ("thinking") tokens at the output rate but reports them separately in
    `thoughts_token_count`, so they are folded in here and also surfaced
    on their own via `thoughts_tokens`.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    thoughts_tokens: int = 0
    audio_input_tokens: int = 0
    cached_tokens: int = 0
    pricing_known: bool = True


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
    truncated: bool = False
    finish_reason: str = ""


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


# Audio variants. An audio file has no slides, so the video prompts' "Visual
# Content" sections and [SLIDE: ...] markers invite the model to invent them.
# These keep the same document shape minus anything visual.
AUDIO_COMPREHENSIVE_PROMPT = """You are an expert content analyst. Analyze this AUDIO recording with extreme thoroughness and produce a comprehensive markdown document.
{duration_line}
# INSTRUCTIONS
- Listen to the ENTIRE recording carefully
- This is AUDIO ONLY. There is no video track, no slides, and nothing on screen
- NEVER describe visual content, slides, or screen text. Do not emit [SLIDE: ...] markers
- Capture EVERY spoken word and all non-speech audio cues (laughter, pauses, tone shifts, background sounds) that carry meaning
- Be extremely detailed - this document should allow someone to fully understand the recording without listening to it
- Use proper markdown formatting throughout

# OUTPUT FORMAT

## Recording Information

| Field | Value |
|-------|-------|
| **Title** | [Descriptive title for this recording] |
| **Duration** | [HH:MM:SS] |
| **Speaker(s)** | [Name(s) and role(s)/title(s) if mentioned] |
| **Organization** | [Company/org if mentioned] |
| **Topic** | [Main subject] |

---

## Executive Summary

[3-5 paragraph comprehensive summary covering what the recording is about, who it's for, main arguments/points made, and key conclusions]

---

## Table of Contents

| Time | Section |
|------|---------|
| [00:00] | Section name |
| [MM:SS] | Section name |

---

## Detailed Content

### [Section Title] [MM:SS - MM:SS]

**Summary**: [2-3 sentence summary of this section]

**Key Points**:
- Point 1
- Point 2

**Transcript Excerpt**:
> "[Important quotes from this section with timestamps]"

[Repeat for each major section]

---

## Full Transcript

[Provide the COMPLETE transcript with timestamps every 30 seconds minimum]

**[00:00]** Speaker: "..."

**[00:30]** Speaker: "..."

[Continue for the entire recording - DO NOT summarize or skip parts]

---

## Key Takeaways

### Main Lessons
1. [Lesson 1 - with brief explanation]
2. [Lesson 2 - with brief explanation]

### Actionable Advice
- [ ] Action item 1
- [ ] Action item 2

### Memorable Quotes
> "[Quote 1]" - [Speaker, MM:SS]

---

## Decisions & Commitments

| Decision / Commitment | Owner | Timestamp |
|-----------------------|-------|-----------|
| [What was decided] | [Who owns it] | [MM:SS] |

---

## Resources Mentioned

| Resource | Type | Reference |
|----------|------|-----------|
| [Name] | [Book/Tool/Website/etc] | [Detail as spoken] |

---

## Glossary

| Term | Definition |
|------|------------|
| [Term used] | [Explanation as given in the recording] |

---

*Analysis generated by Gemini Media Processor*

# CRITICAL REMINDERS
- Capture the FULL transcript, not a summary
- Identify speakers consistently; if names are unknown use stable labels (Speaker 1, Speaker 2)
- Never invent visual content - this is audio only
- Be extremely thorough - more detail is better"""


AUDIO_CONCISE_PROMPT = """Analyze this AUDIO recording. This is audio only: there is no video, no slides, and nothing on screen. Never describe visual content.
{duration_line}
Provide:
1. **Title & Duration**
2. **Speakers** (names and roles if mentioned, otherwise stable labels)
3. **Summary** (2-3 paragraphs)
4. **Key Topics** with timestamps [MM:SS]
5. **Main Takeaways** (bullet points)
6. **Decisions & Action Items** with owners and timestamps where stated

Be comprehensive. Capture the substance of what each speaker contributes."""


AUDIO_TRANSCRIPT_PROMPT = """Provide a complete verbatim transcript of this AUDIO recording.
{duration_line}
Format with timestamps:
[MM:SS] Speaker: "Spoken text..."

Rules:
- This is AUDIO ONLY. Never emit [SLIDE: ...], [DEMO: ...], or any visual marker
- Identify speakers consistently; if names are unknown use stable labels (Speaker 1, Speaker 2)
- Transcribe verbatim, including false starts and filler words
- Note significant non-speech audio in brackets, e.g. [laughter], [long pause], [phone rings]
- Do NOT summarize or skip any part of the recording"""


AUDIO_SEGMENTS_PROMPT = """You are an expert audio analyst. Listen to this ENTIRE recording carefully from start to finish and identify all logical sections/segments.
{duration_line}
# INSTRUCTIONS
- Identify every major topic change, speaker transition, or agenda item boundary
- Focus on semantic/content transitions, not minor pauses
- Each segment should represent a coherent topic or agenda item
- Provide accurate timestamps in HH:MM:SS format
- This is audio only; never reference slides or visual content

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
  }}
]

# CRITICAL RULES
- Return ONLY the JSON array, no other text or markdown
- Cover the ENTIRE recording from start to finish with no gaps
- The LAST segment's end_time MUST match the recording's total duration
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

AUDIO_PROMPTS = {
    "comprehensive": AUDIO_COMPREHENSIVE_PROMPT,
    "concise": AUDIO_CONCISE_PROMPT,
    "transcript": AUDIO_TRANSCRIPT_PROMPT,
    "segments": AUDIO_SEGMENTS_PROMPT,
}


def select_prompt(mode: str, kind: MediaKind = "video") -> str:
    """Return the prompt template for a mode, specialized for audio if needed."""
    if kind == "audio":
        return AUDIO_PROMPTS[mode]
    return PROMPTS[mode]


DEFAULT_MODEL = "gemini-3.1-pro-preview"

# Pricing per 1M tokens (https://ai.google.dev/gemini-api/docs/pricing).
#
# "input"/"output" are the standard per-1M-token rates. "audio_input" is an
# optional override for the portion of input tokens Gemini reports under the
# AUDIO modality, which several models bill at 2-3x the text/video rate.
#
# "long_context_threshold" marks models whose rates change on long prompts.
# This is a CLIFF, not a graduated bracket: Google states the rates as
# "$1.25, prompts <= 200k tokens" / "$2.50, prompts > 200k tokens", so once the
# prompt crosses the threshold every token bills at "long_input", and output
# bills at "long_output" — selected by the PROMPT size, not the output size.
# Billing the first 200k cheaply and only the excess at the high rate would
# under-report the bill on any long-context run.
#
# This table is pricing data only — it is NOT an allow-list. Any model string
# is accepted by --model; unknown models simply report no cost estimate.
MODEL_PRICING: dict[str, dict] = {
    # Pro tier. gemini-3.1-pro-preview is the frontier Pro model as of
    # 2026-08; there is no 3.5/3.6 Pro in the API yet. gemini-2.5-pro is the
    # stable fallback for when a preview model is retired or rate-limited.
    "gemini-3.1-pro-preview": {
        "input": 2.00,
        "output": 12.00,
        "long_context_threshold": 200_000,
        "long_input": 4.00,
        "long_output": 18.00,
    },
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.00,
        "long_context_threshold": 200_000,
        "long_input": 2.50,
        "long_output": 15.00,
    },
    # Flash tier, newest first.
    "gemini-3.6-flash": {"input": 1.50, "output": 7.50},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3-flash-preview": {"input": 0.50, "audio_input": 1.00, "output": 3.00},
    "gemini-3.1-flash-lite": {"input": 0.25, "audio_input": 0.50, "output": 1.50},
    "gemini-2.5-flash": {"input": 0.30, "audio_input": 1.00, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "audio_input": 0.30, "output": 0.40},
}

# Models recommended in --help and README. Ordered best-quality first.
SUGGESTED_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-pro",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]

# Every Gemini 3.x model caps output at 65,536 tokens; 2.5-era models match it.
# Kept as a lookup so a future model with a different cap is a one-line change.
DEFAULT_MAX_OUTPUT_TOKENS = 65536
MODEL_MAX_OUTPUT_TOKENS: dict[str, int] = {}


def get_max_output_tokens(model: str) -> int:
    """Get maximum output tokens for a model."""
    return MODEL_MAX_OUTPUT_TOKENS.get(model, DEFAULT_MAX_OUTPUT_TOKENS)


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


def _rate_cost(rate: float, tokens: int) -> float:
    """Cost of `tokens` at a per-1M-token `rate`."""
    return (tokens / 1_000_000) * rate


def resolve_rates(
    pricing: dict, input_tokens: int
) -> tuple[float, float, float | None]:
    """Pick the input, output, and audio rates for a request.

    Long-context pricing is a cliff, not a graduated bracket. Google's pricing
    page states the rates as "$1.25, prompts <= 200k tokens" / "$2.50, prompts
    > 200k tokens" — so once the PROMPT crosses the threshold, *every* token
    bills at the higher rate, and the output rate is selected by the prompt
    size too (not by how much output was produced).

    Returns:
        (input_rate, output_rate, audio_rate) where audio_rate is None when the
        model has unified input pricing.
    """
    threshold = pricing.get("long_context_threshold")
    is_long = threshold is not None and input_tokens > threshold

    if is_long:
        # No model currently has both a long-context threshold and a separate
        # audio rate; if one ever does, the long rate governs.
        return pricing["long_input"], pricing["long_output"], None

    return pricing["input"], pricing["output"], pricing.get("audio_input")


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    *,
    thoughts_tokens: int = 0,
    audio_input_tokens: int = 0,
    cached_tokens: int = 0,
) -> UsageStats:
    """Calculate usage cost based on model and token counts.

    `output_tokens` is the model's visible answer; `thoughts_tokens` is the
    separately-reported reasoning count. Gemini bills both at the output rate,
    so they are summed into the billed output figure.

    Cached input tokens are billed at a discount, but the discount is not
    published uniformly across models, so they are charged here at the full
    input rate (a conservative over-estimate) and reported separately.

    For an unknown model the token counts are still reported, but every cost
    is zero and `pricing_known` is False — better an obviously-absent number
    than a confidently wrong one from some other model's rate card.
    """
    pricing = MODEL_PRICING.get(model)
    billed_output = output_tokens + thoughts_tokens
    total_tokens = input_tokens + billed_output

    if pricing is None:
        return UsageStats(
            input_tokens=input_tokens,
            output_tokens=billed_output,
            total_tokens=total_tokens,
            thoughts_tokens=thoughts_tokens,
            audio_input_tokens=audio_input_tokens,
            cached_tokens=cached_tokens,
            pricing_known=False,
        )

    input_rate, output_rate, audio_rate = resolve_rates(pricing, input_tokens)

    if audio_rate is not None and audio_input_tokens:
        audio_tokens = min(audio_input_tokens, input_tokens)
        input_cost = _rate_cost(audio_rate, audio_tokens) + _rate_cost(
            input_rate, input_tokens - audio_tokens
        )
    else:
        input_cost = _rate_cost(input_rate, input_tokens)

    output_cost = _rate_cost(output_rate, billed_output)

    return UsageStats(
        input_tokens=input_tokens,
        output_tokens=billed_output,
        total_tokens=total_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        thoughts_tokens=thoughts_tokens,
        audio_input_tokens=audio_input_tokens,
        cached_tokens=cached_tokens,
    )


def _as_int(value) -> int:
    """Coerce a token count off an API response to an int, defaulting to 0.

    Fields are optional and occasionally absent or None depending on model and
    backend; anything not cleanly integral counts as zero rather than
    poisoning the arithmetic downstream.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def extract_usage(response, model: str) -> UsageStats | None:
    """Pull token counts off a response and price them.

    Reads the per-modality breakdown so audio tokens bill at the audio rate,
    and folds `thoughts_token_count` into billed output.
    """
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return None

    input_tokens = _as_int(getattr(usage, "prompt_token_count", 0))
    output_tokens = _as_int(getattr(usage, "candidates_token_count", 0))
    thoughts_tokens = _as_int(getattr(usage, "thoughts_token_count", 0))
    cached_tokens = _as_int(getattr(usage, "cached_content_token_count", 0))

    audio_tokens = 0
    details = getattr(usage, "prompt_tokens_details", None)
    for detail in details if isinstance(details, (list, tuple)) else []:
        modality = getattr(detail, "modality", None)
        name = getattr(modality, "name", None) or str(modality)
        if name and name.upper().endswith("AUDIO"):
            audio_tokens += _as_int(getattr(detail, "token_count", 0))

    return calculate_cost(
        model,
        input_tokens,
        output_tokens,
        thoughts_tokens=thoughts_tokens,
        audio_input_tokens=audio_tokens,
        cached_tokens=cached_tokens,
    )


def _gemini_schema_to_json_schema(schema: dict) -> dict:
    """Recursively lowercase Gemini's UPPERCASE type names to JSON Schema types.

    Gemini's ``response_schema`` uses ``"type": "ARRAY"``/``"OBJECT"``/``"STRING"``
    etc., while the OpenAI-compatible ``response_format.json_schema`` expects
    standard JSON Schema lowercase types (``"array"``, ``"object"``, ...).
    """
    if not isinstance(schema, dict):
        return schema
    result: dict = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            result[key] = value.lower()
        elif key == "items":
            result[key] = _gemini_schema_to_json_schema(value)
        elif key == "properties" and isinstance(value, dict):
            result[key] = {
                k: _gemini_schema_to_json_schema(v) for k, v in value.items()
            }
        else:
            result[key] = value
    return result


def _inline_data_to_openai_part(data: bytes, mime_type: str) -> dict:
    """Map inline media bytes to an OpenAI-compatible content part.

    The Anton/LiteLLM gateway accepts two inline shapes for Gemini models:

    - Audio: ``{"type": "input_audio", "input_audio": {"data": <b64>,
      "format": <fmt>}}`` where ``<fmt>`` is the short subtype (mp3/wav/...).
    - Video: ``{"type": "file", "file": {"file_data":
      "data:<mime>;base64,<b64>"}}``.

    Args:
        data: Raw media bytes.
        mime_type: The media MIME type (e.g. ``audio/mpeg``, ``video/mp4``).
    """
    b64 = base64.b64encode(data).decode("ascii")
    if mime_type.startswith("audio/"):
        audio_format = AUDIO_FORMAT_FOR_MIME.get(mime_type, mime_type.split("/", 1)[-1])
        return {
            "type": "input_audio",
            "input_audio": {"data": b64, "format": audio_format},
        }
    return {
        "type": "file",
        "file": {"file_data": f"data:{mime_type};base64,{b64}"},
    }


class _LiteLLMUsage:
    """Minimal stand-in for genai's ``usage_metadata``."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens


class _LiteLLMResponse:
    """Minimal stand-in for genai's ``GenerateContentResponse``.

    Exposes only what :func:`_call_gemini_and_parse` reads: ``.text`` and
    ``.usage_metadata``.
    """

    def __init__(self, text: str, usage_metadata: _LiteLLMUsage | None) -> None:
        self.text = text
        self.usage_metadata = usage_metadata


class _LiteLLMModels:
    """Implements the ``client.models.generate_content(...)`` surface."""

    def __init__(self, client: LiteLLMClient) -> None:
        self._client = client

    def generate_content(self, *, model: str, contents, config=None):
        """Translate a genai-style request to an OpenAI chat/completions call.

        Only the shapes this tool actually builds are supported: a single
        user ``Content`` whose parts are a media ``Part`` plus a text ``Part``.
        Video/audio URIs (YouTube/GCS) are passed through as an OpenAI ``file``
        content block; local media supplied inline (Gemini ``inline_data`` with
        raw bytes) is emitted as an ``input_audio`` part (audio) or a ``file``
        part with a base64 ``data:`` URL (video).
        """
        # Flatten parts across the provided Content object(s).
        content = contents[0] if isinstance(contents, list) else contents
        parts = getattr(content, "parts", []) or []

        file_uri: str | None = None
        inline_part: dict | None = None
        prompt_text = ""
        has_video_metadata = False
        for part in parts:
            file_data = getattr(part, "file_data", None)
            if file_data is not None and getattr(file_data, "file_uri", None):
                file_uri = file_data.file_uri
                if getattr(part, "video_metadata", None) is not None:
                    has_video_metadata = True
            inline_data = getattr(part, "inline_data", None)
            if inline_data is not None and getattr(inline_data, "data", None):
                inline_part = _inline_data_to_openai_part(
                    inline_data.data, getattr(inline_data, "mime_type", "") or ""
                )
                if getattr(part, "video_metadata", None) is not None:
                    has_video_metadata = True
            text = getattr(part, "text", None)
            if text:
                prompt_text = text

        # Warn about options that have no OpenAI-compatible equivalent.
        media_resolution = getattr(config, "media_resolution", None)
        if has_video_metadata or media_resolution:
            click.echo(
                "Warning: --fps/--clip/--media-resolution are ignored on the "
                "LiteLLM backend (no OpenAI-compatible equivalent).",
                err=True,
            )

        user_content: list[dict] = []
        if prompt_text:
            user_content.append({"type": "text", "text": prompt_text})
        if file_uri:
            user_content.append({"type": "file", "file": {"file_id": file_uri}})
        if inline_part is not None:
            user_content.append(inline_part)

        payload: dict = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
        }

        max_tokens = getattr(config, "max_output_tokens", None)
        if max_tokens:
            payload["max_tokens"] = max_tokens

        response_schema = getattr(config, "response_schema", None)
        if response_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": _gemini_schema_to_json_schema(response_schema),
                },
            }

        resp = self._client.http.post(
            f"{self._client.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._client.api_key}"},
            json=payload,
        )
        if resp.status_code != 200:
            raise click.ClickException(
                f"LiteLLM request failed ({resp.status_code}): {resp.text[:500]}"
            )
        data = resp.json()

        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""

        usage_data = data.get("usage") or {}
        usage = _LiteLLMUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0) or 0,
            completion_tokens=usage_data.get("completion_tokens", 0) or 0,
        )
        return _LiteLLMResponse(text=text, usage_metadata=usage)


class LiteLLMClient:
    """Adapter that speaks an OpenAI-compatible endpoint but duck-types
    :class:`google.genai.Client` closely enough for this tool.

    Only the ``client.models.generate_content(...)`` path is implemented; the
    Gemini Files API (uploads, ``files/*`` refs) is not available over the
    OpenAI-compatible protocol and is guarded against in ``main()``.
    """

    def __init__(self, base_url: str, api_key: str) -> None:
        # Normalize to a bare base (no trailing slash); "/chat/completions"
        # is appended per-request.
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.http = httpx.Client(timeout=httpx.Timeout(600.0))
        self.models = _LiteLLMModels(self)


def _build_litellm_client(base_url: str | None, api_key: str | None) -> LiteLLMClient:
    """Resolve LiteLLM config (arg → env) and construct the client."""
    resolved_base_url = base_url or os.environ.get("LITELLM_BASE_URL")
    if not resolved_base_url:
        raise click.ClickException(
            "LiteLLM backend requires a base URL. Set LITELLM_BASE_URL "
            "environment variable or pass --litellm-base-url"
        )
    resolved_key = api_key or os.environ.get("LITELLM_API_KEY")
    if not resolved_key:
        raise click.ClickException(
            "LiteLLM backend requires an API key. Set LITELLM_API_KEY "
            "environment variable or pass --litellm-api-key"
        )
    click.echo(f"Using LiteLLM endpoint ({resolved_base_url})", err=True)
    return LiteLLMClient(base_url=resolved_base_url, api_key=resolved_key)


def _require_files_api_backend(client) -> None:
    """Reject Files API operations when the LiteLLM backend is active."""
    if isinstance(client, LiteLLMClient):
        raise click.ClickException(
            "The LiteLLM backend does not support the Gemini Files API "
            "(--upload-only / --list-files / --delete-file). Use the Gemini "
            "API key (--api-key) or Vertex AI (--vertex) backend for these."
        )


def get_gemini_client(
    api_key: str | None = None,
    use_vertex: bool = False,
    project: str | None = None,
    location: str | None = None,
    use_litellm: bool = False,
    litellm_base_url: str | None = None,
    litellm_api_key: str | None = None,
):
    """
    Initialize a client with LiteLLM, API key, or Vertex AI authentication.

    Authentication methods (in order of priority):
    1. LiteLLM / OpenAI-compatible endpoint (--litellm flag)
    2. Explicit API key (--api-key flag)
    3. Vertex AI with ADC (--vertex flag) - uses gcloud auth
    4. Auto-detect Vertex AI if GOOGLE_GENAI_USE_VERTEXAI=true
    5. Environment variables (GEMINI_API_KEY or GOOGLE_API_KEY)
    6. Auto-detect LiteLLM if LITELLM_API_KEY is set (last resort)

    The LiteLLM backend targets any OpenAI-compatible gateway that proxies
    Gemini models. It supports YouTube URLs and GCS URIs but NOT the Gemini
    Files API (local uploads, files/* refs) or per-request video metadata
    (fps/clip/media-resolution).
    """
    from google import genai

    litellm_key = litellm_api_key or os.environ.get("LITELLM_API_KEY")

    # An explicit --litellm flag takes top priority over every other backend.
    if use_litellm:
        return _build_litellm_client(litellm_base_url, litellm_key)

    # Check if Vertex AI mode is requested or auto-detected
    vertex_env = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
    use_vertex = use_vertex or vertex_env

    if use_vertex:
        # Vertex AI authentication using Application Default Credentials
        # Requires: gcloud auth application-default login
        gcp_project = project or os.environ.get("YT_PROCESS_PROJECT")
        gcp_location = location or os.environ.get("YT_PROCESS_LOCATION") or "global"

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
    if key:
        return genai.Client(api_key=key)

    # Last resort: auto-detect the LiteLLM backend from LITELLM_API_KEY.
    if litellm_key:
        return _build_litellm_client(litellm_base_url, litellm_key)

    raise click.ClickException(
        "Authentication required. Choose one:\n"
        "  1. API key: Set GEMINI_API_KEY env var or pass --api-key\n"
        "  2. Vertex AI: Pass --vertex flag (requires gcloud auth application-default login)\n"
        "  3. LiteLLM: Set LITELLM_BASE_URL + LITELLM_API_KEY or pass --litellm\n"
        "\nGet an API key at: https://aistudio.google.com/app/apikey"
    )


def validate_youtube_url(url: str) -> str:
    """Validate and normalize YouTube URL."""
    patterns = [
        r"(?:https?://)?(?:www\.|m\.)?youtube\.com/watch\?(?:[^#]*&)?v=([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.|m\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)",
        r"(?:https?://)?(?:www\.|m\.)?youtube\.com/shorts/([a-zA-Z0-9_-]+)",
        # Premiere/stream replays are served under /live/
        r"(?:https?://)?(?:www\.|m\.)?youtube\.com/live/([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            return f"https://www.youtube.com/watch?v={video_id}"

    raise click.ClickException(f"Invalid YouTube URL: {url}")


def extract_video_id(url: str) -> str | None:
    """Extract video ID from a YouTube URL in any of its supported shapes."""
    for pattern in (
        r"[?&]v=([a-zA-Z0-9_-]+)",
        r"youtu\.be/([a-zA-Z0-9_-]+)",
        r"youtube\.com/(?:embed|shorts|live)/([a-zA-Z0-9_-]+)",
    ):
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _normalize_timestamp_to_hhmmss(timestamp: str) -> str:
    """Normalize a MM:SS or HH:MM:SS timestamp to HH:MM:SS format."""
    parts = timestamp.split(":")
    if len(parts) == 2:
        return f"00:{parts[0].zfill(2)}:{parts[1]}"
    if len(parts) == 3:
        return f"{parts[0].zfill(2)}:{parts[1]}:{parts[2]}"
    return timestamp


@lru_cache(maxsize=8)
def _fetch_youtube_page(video_id: str) -> str:
    """Fetch a YouTube watch page's HTML, or "" on any failure.

    Cached so chapter and duration lookups for the same video share one
    request.
    """
    fetch_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        req = urllib.request.Request(
            fetch_url,
            headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _decode_json_string(raw: str) -> str:
    """Decode a JSON string body scraped out of YouTube's embedded page data.

    The value still carries JSON escapes (\\n, \\u00e9). Decoding it as a JSON
    string is correct; the older `.encode().decode("unicode_escape")` trick
    mangles any non-ASCII text into mojibake ("Café" -> "CafÃ©").
    """
    try:
        return json.loads(f'"{raw}"')
    except json.JSONDecodeError:
        return raw


def fetch_youtube_duration(url: str) -> str | None:
    """Fetch a YouTube video's duration as HH:MM:SS, or None if unavailable.

    Reads `lengthSeconds` out of the watch page, which is the same page the
    chapter scrape already fetches.
    """
    video_id = extract_video_id(url)
    if not video_id:
        return None

    html = _fetch_youtube_page(video_id)
    match = re.search(r'"lengthSeconds":"(\d+)"', html)
    if not match:
        return None
    return _format_duration(float(match.group(1)))


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

    html = _fetch_youtube_page(video_id)
    if not html:
        return []

    # Extract description from page data
    description = ""
    # Match a full JSON string body, honoring backslash escapes, so a
    # description containing an escaped quote is not truncated at it.
    for pattern in [
        r'"shortDescription":"((?:[^"\\]|\\.)*)"',
        r'"description":\{"simpleText":"((?:[^"\\]|\\.)*)"\}',
    ]:
        match = re.search(pattern, html)
        if match:
            description = _decode_json_string(match.group(1))
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

# Map audio MIME types to the short "format" token expected by the OpenAI
# ``input_audio`` content part. Covers the values in AUDIO_MIME_TYPES plus a few
# common aliases (audio/mp3, audio/x-m4a, audio/x-wav, audio/x-flac).
AUDIO_FORMAT_FOR_MIME = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/ogg": "ogg",
    "audio/aiff": "aiff",
}


def is_local_file(input_path: str) -> bool:
    """Check if input is a local file path (exists on disk)."""
    path = Path(input_path)
    return path.exists() and path.is_file()


def is_youtube_url(url: str) -> bool:
    """Check if input looks like a YouTube URL."""
    youtube_patterns = [
        r"(?:https?://)?(?:www\.|m\.)?youtube\.com",
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


def _build_video_metadata(
    kind: MediaKind,
    fps: float | None,
    clip_start: str | None,
    clip_end: str | None,
):
    """Build VideoMetadata for a media part, or None if nothing to attach.

    `fps` is video-only; clip offsets apply to both audio and video.
    """
    from google.genai import types

    vm_kwargs: dict = {}
    if kind == "video" and fps is not None:
        vm_kwargs["fps"] = fps
    if clip_start is not None:
        vm_kwargs["start_offset"] = clip_start
    if clip_end is not None:
        vm_kwargs["end_offset"] = clip_end

    return types.VideoMetadata(**vm_kwargs) if vm_kwargs else None


def build_media_part(
    file_uri: str,
    mime_type: str,
    *,
    kind: MediaKind = "video",
    fps: float | None = None,
    clip_start: str | None = None,
    clip_end: str | None = None,
):
    """Build a media Part referencing a remote URI, with optional VideoMetadata.

    Args:
        file_uri: The file URI (Files API, GCS, or YouTube URL).
        mime_type: MIME type of the media.
        kind: Media kind, "video" or "audio". Audio never gets an fps value.
        fps: Custom frames per second for sampling (video only).
        clip_start: Start offset in "{seconds}s" format.
        clip_end: End offset in "{seconds}s" format.
    """
    from google.genai import types

    part_kwargs: dict = {
        "file_data": types.FileData(file_uri=file_uri, mime_type=mime_type),
    }
    video_metadata = _build_video_metadata(kind, fps, clip_start, clip_end)
    if video_metadata:
        part_kwargs["video_metadata"] = video_metadata

    return types.Part(**part_kwargs)


def build_inline_media_part(
    data: bytes,
    mime_type: str,
    *,
    kind: MediaKind = "video",
    fps: float | None = None,
    clip_start: str | None = None,
    clip_end: str | None = None,
):
    """Build a media Part carrying the file bytes inline.

    Used for the Vertex AI backend, which has no Files API. The bytes ride
    along in the request, so this is only viable below INLINE_MAX_BYTES.
    """
    from google.genai import types

    part_kwargs: dict = {
        "inline_data": types.Blob(data=data, mime_type=mime_type),
    }
    video_metadata = _build_video_metadata(kind, fps, clip_start, clip_end)
    if video_metadata:
        part_kwargs["video_metadata"] = video_metadata

    return types.Part(**part_kwargs)


# Vertex has no Files API, so local files ride inline in the request. Google's
# documented ceiling for total request size is 20 MB; past that the file has to
# go to GCS first.
INLINE_MAX_BYTES = 20 * 1024 * 1024


def is_vertex_client(client) -> bool:
    """Return True if the client is talking to the Vertex AI backend.

    Compares against True rather than testing truthiness: the SDK sets a real
    bool, and anything else (a stub, a mock, an unset attribute) should read
    as "not Vertex" rather than silently taking the Vertex path.
    """
    return getattr(client, "vertexai", False) is True


def _require_developer_api(client, feature: str) -> None:
    """Raise a clear error when a Files-API-only feature is used on Vertex.

    The SDK's own error ("This method is only supported in the Gemini Developer
    client") gives no hint about what to do instead.
    """
    if is_vertex_client(client):
        raise click.ClickException(
            f"{feature} uses the Files API, which Vertex AI does not provide.\n"
            "Either drop --vertex and authenticate with GEMINI_API_KEY, or pass "
            "a gs:// URI (see --gcs-bucket) instead of a local path."
        )


def gcs_object_name(local_path: Path) -> str:
    """Build a collision-free object name for a staged local file.

    The bare basename is not safe: a batch containing two different
    `recording.m4a` files from different directories would map both to the
    same object, and `gcloud storage cp` overwrites silently — so one input
    would be analyzed against the other's bytes. Prefixing with a short digest
    of the absolute path keeps distinct sources distinct while staying
    deterministic, so re-running the same file reuses one object instead of
    littering the bucket.
    """
    digest = hashlib.sha256(str(local_path).encode("utf-8")).hexdigest()[:12]
    return f"yt-process/{digest}/{local_path.name}"


def upload_to_gcs(local_path: Path, bucket: str, verbose: bool = False) -> str:
    """Upload a local file to GCS via the gcloud CLI and return its gs:// URI.

    Shells out to `gcloud storage cp` so the tool inherits the same ADC
    session Vertex already uses, and so google-cloud-storage stays out of the
    dependency set.

    The uploaded object PERSISTS. Nothing in this tool deletes it, and unlike
    the Files API (48h expiry) GCS has no default TTL, so staging a
    confidential recording leaves a permanent plaintext copy in the bucket.
    Callers should prefer the inline path whenever the file is small enough.
    """
    if not shutil.which("gcloud"):
        raise click.ClickException(
            "--gcs-bucket requires the gcloud CLI, which was not found on PATH."
        )

    prefix = bucket.removeprefix("gs://").strip("/")
    if not prefix:
        raise click.ClickException(f"--gcs-bucket is not a usable bucket: {bucket!r}")

    gcs_uri = f"gs://{prefix}/{gcs_object_name(local_path)}"

    if verbose:
        click.echo(f"  Uploading {local_path.name} to {gcs_uri}...", err=True)

    result = subprocess.run(
        ["gcloud", "storage", "cp", str(local_path), gcs_uri],
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"Upload to {gcs_uri} failed: {result.stderr.strip()[:400]}"
        )

    click.echo(
        f"  Uploaded (persists until you delete it): {gcs_uri}",
        err=True,
    )
    return gcs_uri


def build_generate_config(
    model: str,
    *,
    response_schema: dict | None = None,
    media_resolution: str | None = None,
    thinking_level: str | None = None,
):
    """Build a GenerateContentConfig with optional media resolution.

    Args:
        model: Model name (used to determine max output tokens).
        response_schema: Optional JSON schema for structured output.
        media_resolution: One of "low", "medium", "high", or None.
        thinking_level: One of "minimal", "low", "medium", "high", or None.
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
    if thinking_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level.upper()
        )

    return types.GenerateContentConfig(**config_kwargs)


# HTTP status codes worth retrying: rate limits and transient server faults.
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
RETRY_BASE_DELAY_SECONDS = 2.0


def _is_retryable(exc: Exception) -> bool:
    """Return True if an API exception is worth retrying."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        # An explicit status code is authoritative in both directions: a 400
        # whose message happens to contain the word "UNAVAILABLE" must not be
        # retried, and retrying re-sends the whole inline payload.
        return code in RETRYABLE_STATUS_CODES

    # Fall back to the status string only when no code is exposed, and anchor
    # on the canonical "<CODE> <STATUS>" prefix the SDK emits rather than
    # searching the entire message body.
    text = str(exc).upper()
    return any(
        re.search(rf"\b(?:{code_re})\b\s+{marker}", text)
        for code_re, marker in (
            (r"429", "RESOURCE_EXHAUSTED"),
            (r"503", "UNAVAILABLE"),
            (r"504", "DEADLINE_EXCEEDED"),
            (r"500", "INTERNAL"),
        )
    )


def generate_with_retry(
    client,
    *,
    model: str,
    contents,
    config,
    max_retries: int = 3,
    verbose: bool = False,
    sleep=time.sleep,
):
    """Call generate_content, retrying rate limits and transient server errors.

    Uses exponential backoff (2s, 4s, 8s, ...). Non-retryable errors and the
    final attempt's error propagate to the caller unchanged.
    """
    attempt = 0
    while True:
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:
            if attempt >= max_retries or not _is_retryable(exc):
                raise
            # Jitter keeps parallel workers that hit the same quota wall from
            # retrying in lockstep.
            delay = RETRY_BASE_DELAY_SECONDS * (2**attempt)
            delay *= random.uniform(0.75, 1.25)  # noqa: S311 - backoff, not crypto
            attempt += 1
            if verbose:
                click.echo(
                    f"  Retry {attempt}/{max_retries} in {delay:.0f}s ({exc})",
                    err=True,
                )
            sleep(delay)


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
    except (ValueError, OSError, subprocess.SubprocessError):
        # TimeoutExpired is a SubprocessError, NOT an OSError: without it a slow
        # ffprobe escapes this helper and aborts the whole run.
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
    except (ValueError, OSError, subprocess.SubprocessError):
        # TimeoutExpired is a SubprocessError, NOT an OSError: without it a slow
        # ffprobe escapes this helper and aborts the whole run.
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


def extract_title(text: str, fallback_title: str = "") -> str:
    """Pull a title out of a model response.

    Tries the structured forms the prompts actually ask for before falling
    back to anything loosely resembling a title, so a stray "Video
    Information" heading no longer wins over the real title.
    """
    patterns = [
        # | **Title** | Real Title |  (the comprehensive prompt's table)
        r"^\|\s*\*{0,2}Title\*{0,2}\s*\|\s*(.+?)\s*\|",
        # **Title**: Real Title  /  Title: Real Title
        r"^\s*\*{0,2}Title\*{0,2}\s*:\s*(.+?)\s*$",
        # # Real Title  (a level-1 heading that is not our own boilerplate)
        r"^#\s+(?!Video Analysis|Media Analysis|Audio Analysis|Error )(.+?)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            title = match.group(1).strip().strip("*_`\"'[]")
            # Reject the prompt's own placeholder text.
            if title and not title.lower().startswith("full video title"):
                return title
    return fallback_title


def _call_gemini_and_parse(
    client,
    media_part,
    model: str,
    prompt: str,
    analysis: VideoAnalysis,
    fallback_title: str = "",
    response_schema: dict | None = None,
    media_resolution: str | None = None,
    thinking_level: str | None = None,
    max_retries: int = 3,
    verbose: bool = False,
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
        thinking_level: Optional reasoning effort setting.
        max_retries: Retry budget for rate limits and transient server errors.
        verbose: Print retry progress.
    """
    from google.genai import types

    config = build_generate_config(
        model,
        response_schema=response_schema,
        media_resolution=media_resolution,
        thinking_level=thinking_level,
    )

    response = generate_with_retry(
        client,
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=[media_part, types.Part(text=prompt)],
            )
        ],
        config=config,
        max_retries=max_retries,
        verbose=verbose,
    )

    analysis.usage = extract_usage(response, model)

    # A candidate can finish without usable text: blocked by a safety filter,
    # or the token budget spent entirely on reasoning. response.text is None
    # in that case, which would otherwise blow up the regexes below.
    finish_reason = ""
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        raw_reason = getattr(candidates[0], "finish_reason", None)
        if raw_reason is not None:
            # Prefer the enum's name; fall back to str() so the field is always
            # a plain string and stays JSON-serializable.
            name = getattr(raw_reason, "name", None)
            finish_reason = name if isinstance(name, str) else str(raw_reason)
    analysis.finish_reason = finish_reason

    text = response.text
    if text is None:
        detail = f" (finish_reason: {finish_reason})" if finish_reason else ""
        feedback = getattr(response, "prompt_feedback", None)
        if feedback:
            detail += f" (prompt_feedback: {feedback})"
        raise click.ClickException(
            f"Model returned no text content{detail}. "
            "The response may have been blocked, or the output budget was "
            "consumed by reasoning — try --thinking-level low or a shorter --clip."
        )

    analysis.raw_response = text

    # MAX_TOKENS means the document is cut off mid-sentence. Surfacing this is
    # the difference between a knowingly-partial transcript and a silently
    # wrong one.
    if finish_reason == "MAX_TOKENS":
        analysis.truncated = True
        click.echo(
            f"  WARNING: output hit the {get_max_output_tokens(model):,}-token cap "
            "and is TRUNCATED. Re-run with --clip to process a shorter range, "
            "or --mode transcript for a less verbose format.",
            err=True,
        )

    analysis.title = extract_title(text, fallback_title)

    # Extract summary if present
    summary_match = re.search(
        r"(?:## (?:5\. )?summary|summary:)\s*\n(.*?)(?=\n## |\n\*\*|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if summary_match:
        analysis.summary = summary_match.group(1).strip()


def process_local_file(
    client,
    file_path: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    response_schema: dict | None = None,
    fps: float | None = None,
    clip_start: str | None = None,
    clip_end: str | None = None,
    media_resolution: str | None = None,
    thinking_level: str | None = None,
    max_retries: int = 3,
    gcs_bucket: str | None = None,
) -> VideoAnalysis:
    """Process a local media file with the Gemini API.

    Transport depends on the backend. The Gemini Developer API uses the Files
    API. Vertex AI has no Files API, so the bytes go inline when small enough
    and via GCS otherwise.
    """
    path = Path(file_path).resolve()

    analysis = VideoAnalysis(
        url=str(path),
        processed_at=datetime.now().isoformat(),
        model=model,
    )

    try:
        mime_type, kind = get_media_mime_type(path)
        file_size = path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        # LiteLLM has no Files API: send the bytes inline as a Gemini
        # inline_data Part, which the LiteLLM adapter maps to the
        # OpenAI-compatible audio/video content shapes.
        if isinstance(client, LiteLLMClient):
            if fps is not None or clip_start is not None or clip_end is not None:
                click.echo(
                    "Warning: --fps/--clip are ignored on the LiteLLM backend "
                    "(no OpenAI-compatible equivalent).",
                    err=True,
                )
            if verbose:
                click.echo(
                    f"  Sending {path.name} ({file_size_mb:.1f} MB) inline...",
                    err=True,
                )
            media_part = build_inline_media_part(
                path.read_bytes(), mime_type, kind=kind
            )
        elif is_vertex_client(client):
            media_part = _build_vertex_local_part(
                path,
                mime_type,
                kind,
                file_size=file_size,
                gcs_bucket=gcs_bucket,
                verbose=verbose,
                fps=fps,
                clip_start=clip_start,
                clip_end=clip_end,
            )
        else:
            if verbose:
                click.echo(
                    f"  Uploading {path.name} ({file_size_mb:.1f} MB)...", err=True
                )

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
                raise click.ClickException(
                    f"File processing failed: {uploaded_file.name}"
                )

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
            thinking_level=thinking_level,
            max_retries=max_retries,
            verbose=verbose,
        )

    except Exception as e:
        analysis.error = str(e)

    return analysis


def _build_vertex_local_part(
    path: Path,
    mime_type: str,
    kind: MediaKind,
    *,
    file_size: int,
    gcs_bucket: str | None,
    verbose: bool,
    fps: float | None,
    clip_start: str | None,
    clip_end: str | None,
):
    """Build a media part for a local file on the Vertex backend.

    Small files ride inline in the request. Larger ones are staged to GCS
    when a bucket is configured; otherwise this raises with both options
    spelled out, because the SDK's own error explains neither.

    Inline is preferred even when a bucket is configured. Staging leaves a
    permanent copy of the media in cloud storage, so a file small enough to
    ride along in the request should never be persisted as a side effect of
    having set YT_PROCESS_GCS_BUCKET once.
    """
    file_size_mb = file_size / (1024 * 1024)

    if file_size <= INLINE_MAX_BYTES:
        if verbose:
            click.echo(
                f"  Inlining {path.name} ({file_size_mb:.1f} MB) for Vertex...",
                err=True,
            )
        return build_inline_media_part(
            path.read_bytes(),
            mime_type,
            kind=kind,
            fps=fps,
            clip_start=clip_start,
            clip_end=clip_end,
        )

    if gcs_bucket:
        gcs_uri = upload_to_gcs(path, gcs_bucket, verbose=verbose)
        return build_media_part(
            gcs_uri,
            mime_type,
            kind=kind,
            fps=fps,
            clip_start=clip_start,
            clip_end=clip_end,
        )

    raise click.ClickException(
        f"{path.name} is {file_size_mb:.1f} MB, over the "
        f"{INLINE_MAX_BYTES // (1024 * 1024)} MB inline limit, and Vertex AI has "
        "no Files API. Either:\n"
        "  1. Stage it automatically: --gcs-bucket YOUR_BUCKET\n"
        "  2. Upload it yourself and pass the gs:// URI as input\n"
        "  3. Process a smaller range with --clip\n"
        "  4. Drop --vertex and use GEMINI_API_KEY (Files API handles 2 GB)"
    )


def process_files_api_ref(
    client,
    file_ref: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    response_schema: dict | None = None,
    fps: float | None = None,
    clip_start: str | None = None,
    clip_end: str | None = None,
    media_resolution: str | None = None,
    thinking_level: str | None = None,
    max_retries: int = 3,
) -> VideoAnalysis:
    """Process a video using an existing Files API reference.

    Skips upload entirely and uses a previously uploaded file.
    Files API references expire after 48 hours.
    """
    _require_developer_api(client, "A files/ reference")

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
            thinking_level=thinking_level,
            max_retries=max_retries,
            verbose=verbose,
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
    model: str = DEFAULT_MODEL,
    verbose: bool = False,
    response_schema: dict | None = None,
    fps: float | None = None,
    clip_start: str | None = None,
    clip_end: str | None = None,
    media_resolution: str | None = None,
    thinking_level: str | None = None,
    max_retries: int = 3,
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
            thinking_level=thinking_level,
            max_retries=max_retries,
            verbose=verbose,
        )

    except Exception as e:
        analysis.error = str(e)

    return analysis


def process_video(
    client,
    url: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    response_schema: dict | None = None,
    fps: float | None = None,
    clip_start: str | None = None,
    clip_end: str | None = None,
    media_resolution: str | None = None,
    thinking_level: str | None = None,
    max_retries: int = 3,
    verbose: bool = False,
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
            thinking_level=thinking_level,
            max_retries=max_retries,
            verbose=verbose,
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

    usage_section = format_usage_markdown(analysis.usage)

    lines = [
        "# Video Segments Analysis",
        "",
        f"**Source**: {analysis.url}",
        f"**Processed**: {analysis.processed_at}",
        f"**Model**: {analysis.model}{usage_section}{_truncation_banner(analysis)}",
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
    return json.dumps(
        {
            "url": analysis.url,
            "processed_at": analysis.processed_at,
            "model": analysis.model,
            "segments": segments,
            "usage": format_usage_dict(analysis.usage),
            "truncated": analysis.truncated,
            "finish_reason": analysis.finish_reason,
            "error": analysis.error,
        },
        indent=2,
    )


def media_kind_for_input(input_str: str) -> MediaKind:
    """Classify an input as audio or video by its extension.

    YouTube URLs and files/ references have no usable extension, so they fall
    back to "video"; for a files/ reference the true kind is still resolved
    from the API metadata when building the media part.
    """
    return "audio" if is_audio_input(input_str) else "video"


def build_duration_line(
    video_input: str,
    *,
    clip_start: str | None = None,
    clip_end: str | None = None,
    verbose: bool = False,
) -> str:
    """Build the prompt fragment that tells the model how long the media is.

    Coverage improves measurably when the model knows the target length. With
    --clip the model only sees the requested range, so the clip's length is
    described instead of the source's; announcing the full duration there
    would ask it to cover time it was never shown.
    """
    if clip_start is not None or clip_end is not None:
        if clip_start is None or clip_end is None:
            return ""
        start = int(clip_start.rstrip("s"))
        end = int(clip_end.rstrip("s"))
        if end <= start:
            return ""
        span = _format_duration(end - start)
        return (
            f"\nThis is a {span}-long excerpt. "
            f"You MUST cover the entire excerpt from start to finish.\n"
        )

    duration = None
    if is_local_file(video_input):
        duration = get_video_duration(video_input)
    elif is_gcs_uri(video_input):
        duration = get_video_duration_gcs(video_input)
    elif is_youtube_url(video_input):
        duration = fetch_youtube_duration(video_input)

    if not duration:
        return ""

    if verbose:
        click.echo(f"  Detected media duration: {duration}", err=True)
    return (
        f"\nThe media is exactly {duration} long. "
        f"You MUST cover from 00:00:00 to {duration}.\n"
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
    thinking_level: str | None = None,
    max_retries: int = 3,
    is_template: bool = True,
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

    # The built-in prompts carry a {duration_line} placeholder. It has to be
    # filled here, per chapter, or the literal "{duration_line}" string is what
    # reaches the model. A custom --prompt is passed through untouched because
    # user text may contain braces of its own.
    chapter_prompt = analysis_prompt
    if is_template:
        span = ""
        if clip_end:
            seconds = int(clip_end.rstrip("s")) - int(clip_start.rstrip("s"))
            if seconds > 0:
                span = (
                    f"\nThis is a {_format_duration(seconds)}-long excerpt "
                    f"beginning at {start} of a longer video. You MUST cover the "
                    f"entire excerpt.\n"
                )
        chapter_prompt = analysis_prompt.format(duration_line=span)

    try:
        analysis = process_video(
            client,
            url,
            chapter_prompt,
            model,
            response_schema=None,
            fps=fps,
            clip_start=clip_start,
            clip_end=clip_end,
            media_resolution=media_resolution,
            thinking_level=thinking_level,
            max_retries=max_retries,
            verbose=verbose,
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
    out_path.write_text(formatted, encoding="utf-8")

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
    thinking_level: str | None = None,
    max_retries: int = 3,
    is_template: bool = True,
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
                thinking_level,
                max_retries,
                is_template,
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


def format_usage_markdown(usage: UsageStats | None) -> str:
    """Render a usage/cost block for a markdown document."""
    if not usage:
        return ""

    breakdown = f"{usage.input_tokens:,} input"
    if usage.audio_input_tokens:
        breakdown += f" ({usage.audio_input_tokens:,} audio)"
    breakdown += f" + {usage.output_tokens:,} output"
    if usage.thoughts_tokens:
        breakdown += f" (incl. {usage.thoughts_tokens:,} thinking)"
    breakdown += f" = {usage.total_tokens:,} tokens"
    if usage.cached_tokens:
        breakdown += f"; {usage.cached_tokens:,} cached"

    if not usage.pricing_known:
        return f"\n**Usage**: {breakdown}\n**Cost**: unknown (no published pricing for this model)\n"

    return (
        f"\n**Usage**: {breakdown}\n"
        f"**Cost**: ${usage.total_cost:.6f} "
        f"(${usage.input_cost:.6f} input + ${usage.output_cost:.6f} output)\n"
    )


def format_usage_dict(usage: UsageStats | None) -> dict | None:
    """Render a usage/cost block for JSON output."""
    if not usage:
        return None
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "thoughts_tokens": usage.thoughts_tokens,
        "audio_input_tokens": usage.audio_input_tokens,
        "cached_tokens": usage.cached_tokens,
        "input_cost_usd": usage.input_cost if usage.pricing_known else None,
        "output_cost_usd": usage.output_cost if usage.pricing_known else None,
        "total_cost_usd": usage.total_cost if usage.pricing_known else None,
        "pricing_known": usage.pricing_known,
    }


def _truncation_banner(analysis: VideoAnalysis) -> str:
    """Render an in-document warning when the model's output was cut off."""
    if not analysis.truncated:
        return ""
    return (
        "\n> **WARNING: TRUNCATED OUTPUT.** The model hit its output token cap, "
        "so this document is incomplete and stops mid-content. Re-run with "
        "`--clip` over a narrower range, or `--mode transcript`, to get full "
        "coverage.\n"
    )


def format_output_markdown(analysis: VideoAnalysis, kind: MediaKind = "video") -> str:
    """Format analysis as markdown."""
    label = "Audio" if kind == "audio" else "Video"
    if analysis.error:
        return f"# Error Processing {label}\n\n**URL**: {analysis.url}\n\n**Error**: {analysis.error}\n"

    return f"""# {label} Analysis

**URL**: {analysis.url}
**Processed**: {analysis.processed_at}
**Model**: {analysis.model}{format_usage_markdown(analysis.usage)}{_truncation_banner(analysis)}

---

{analysis.raw_response}
"""


def format_output_json(analysis: VideoAnalysis, kind: MediaKind = "video") -> str:
    """Format analysis as JSON."""
    return json.dumps(
        {
            "url": analysis.url,
            "title": analysis.title,
            "processed_at": analysis.processed_at,
            "model": analysis.model,
            "media_kind": kind,
            "content": analysis.raw_response,
            "summary": analysis.summary,
            "usage": format_usage_dict(analysis.usage),
            "truncated": analysis.truncated,
            "finish_reason": analysis.finish_reason,
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
    _require_developer_api(
        client, "--list-files/--delete-file" if list_files else "--delete-file"
    )
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
    _require_developer_api(client, "--upload-only")

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
    thinking_level: str | None = None,
    max_retries: int = 3,
    is_template: bool = True,
) -> int:
    """Handle YouTube chapter-based splitting (--split without segments mode).

    Returns the number of chapters that failed to process.
    """
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
        thinking_level=thinking_level,
        max_retries=max_retries,
        is_template=is_template,
    )

    click.echo(f"\nCreated {len(created)} chapter files in {split_out}", err=True)
    return len(chapters) - len(created)


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
        file_path.write_text(formatted, encoding="utf-8")
        if verbose:
            click.echo(f"  Saved to: {file_path}")
    elif output_file and not is_batch:
        # Single file output
        output_file.write_text(formatted, encoding="utf-8")
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
    type=str,
    default=DEFAULT_MODEL,
    help=f"Gemini model to use (default: {DEFAULT_MODEL}). Any model ID is "
    f"accepted; known models are {', '.join(SUGGESTED_MODELS)}.",
)
@click.option(
    "--thinking-level",
    type=click.Choice(["minimal", "low", "medium", "high"], case_sensitive=False),
    default=None,
    help="Reasoning effort for Gemini 3 models. Lower is cheaper and faster; "
    "thinking tokens bill at the output rate.",
)
@click.option(
    "--timestamp-offset",
    type=str,
    default=None,
    help="Shift emitted timestamps by this offset so a chunk's output reads in "
    "the original recording's timeline. Format: SS, MM:SS, or HH:MM:SS.",
)
@click.option(
    "--gcs-bucket",
    type=str,
    envvar="YT_PROCESS_GCS_BUCKET",
    default=None,
    help="GCS bucket for staging local files on Vertex AI, which has no Files "
    "API (or set YT_PROCESS_GCS_BUCKET). Requires the gcloud CLI.",
)
@click.option(
    "--max-retries",
    type=click.IntRange(0, 10),
    default=3,
    help="Retries for rate limits and transient server errors (default: 3).",
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
    "--litellm",
    "use_litellm",
    is_flag=True,
    help="Use a LiteLLM / OpenAI-compatible endpoint (set LITELLM_BASE_URL and "
    "LITELLM_API_KEY). Supports YouTube URLs and GCS URIs only.",
)
@click.option(
    "--litellm-base-url",
    envvar="LITELLM_BASE_URL",
    default=None,
    help="Base URL for the LiteLLM endpoint, e.g. https://host/v1 "
    "(or set LITELLM_BASE_URL)",
)
@click.option(
    "--litellm-api-key",
    envvar="LITELLM_API_KEY",
    default=None,
    help="API key for the LiteLLM endpoint (or set LITELLM_API_KEY)",
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
    type=click.IntRange(1, 32),
    default=4,
    help="Parallel workers for --split and for batch mode writing to a "
    "directory (default: 4). Each in-flight local file is held in memory.",
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
    thinking_level: str | None,
    timestamp_offset: str | None,
    gcs_bucket: str | None,
    max_retries: int,
    api_key: str | None,
    vertex: bool,
    project: str | None,
    location: str | None,
    use_litellm: bool,
    litellm_base_url: str | None,
    litellm_api_key: str | None,
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

        Option 3 - LiteLLM / OpenAI-compatible endpoint:
            export LITELLM_BASE_URL="https://your-gateway/v1"
            export LITELLM_API_KEY="sk-..."
            yt-process "URL" --litellm
            (YouTube URLs and GCS URIs only; no Files API / local uploads)

    \b
    Environment Variables:
        GEMINI_API_KEY           Google Gemini API key
        GOOGLE_API_KEY           Alternative API key variable
        YT_PROCESS_PROJECT       GCP project for Vertex AI (required with --vertex)
        YT_PROCESS_LOCATION      GCP location for Vertex AI (default: global)
        GOOGLE_GENAI_USE_VERTEXAI  Set to "true" to auto-enable Vertex AI
        LITELLM_BASE_URL         Base URL for a LiteLLM / OpenAI-compatible endpoint
        LITELLM_API_KEY          API key for the LiteLLM endpoint
        YT_PROCESS_GCS_BUCKET    GCS bucket for staging local files on Vertex

    \b
    Vertex AI and local files:
        Vertex has no Files API. Files under 20 MB are sent inline
        automatically; larger ones need a staging bucket:
            yt-process ./long-meeting.m4a --vertex --gcs-bucket my-bucket
    """
    # Auto-detect location based on model if not specified
    if location is None:
        # Default to global for all models to avoid regional quota limits
        location = "global"

    if model not in MODEL_PRICING:
        click.echo(
            f"Note: '{model}' has no entry in the pricing table, so cost will be "
            f"reported as unknown. Known models: {', '.join(SUGGESTED_MODELS)}",
            err=True,
        )

    # Parse video processing options
    clip_start: str | None = None
    clip_end: str | None = None
    if clip:
        clip_start, clip_end = parse_clip_range(clip)

    resolved_media_resolution: str | None = None
    if media_resolution:
        resolved_media_resolution = MEDIA_RESOLUTION_MAP[media_resolution]

    offset_line = ""
    if timestamp_offset:
        offset_seconds = int(parse_timestamp_to_seconds(timestamp_offset).rstrip("s"))
        offset_hhmmss = _format_duration(offset_seconds)
        offset_line = (
            f"\nThis media is an excerpt beginning at {offset_hhmmss} of a longer "
            f"recording. Add {offset_hhmmss} to every timestamp you emit so all "
            f"timestamps refer to the original recording's timeline. The first "
            f"moment of this excerpt is {offset_hhmmss}, not 00:00:00.\n"
        )

    # Handle file management operations (no INPUT required)
    if list_files or delete_file:
        client = get_gemini_client(
            api_key=api_key,
            use_vertex=vertex,
            project=project,
            location=location,
            use_litellm=use_litellm,
            litellm_base_url=litellm_base_url,
            litellm_api_key=litellm_api_key,
        )
        _require_files_api_backend(client)
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
        use_litellm=use_litellm,
        litellm_base_url=litellm_base_url,
        litellm_api_key=litellm_api_key,
    )

    # Handle upload-only mode
    if upload_only:
        if not input:
            raise click.ClickException(
                "--upload-only requires a local file path as input"
            )
        _require_files_api_backend(client)
        _handle_upload_only(client, input, verbose)
        return

    # Determine prompt
    is_segments_mode = mode == "segments" and not prompt
    analysis_prompt = prompt if prompt else PROMPTS[mode]
    if not prompt and input and media_kind_for_input(input) == "audio":
        analysis_prompt = select_prompt(mode, "audio")

    # Collect inputs to process (URLs or file paths)
    inputs: list[str] = []
    if batch:
        batch_path = Path(batch)
        inputs = [
            line.strip()
            for line in batch_path.read_text(encoding="utf-8").splitlines()
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
        if isinstance(client, LiteLLMClient):
            raise click.ClickException(
                "Chapter splitting (--split) relies on per-chapter clipping, "
                "which the LiteLLM backend does not support. Use the Gemini API "
                "key (--api-key) or Vertex AI (--vertex) backend for --split."
            )
        chapter_failures = _handle_chapter_splitting(
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
            thinking_level=thinking_level,
            max_retries=max_retries,
            is_template=not prompt,
        )
        if chapter_failures:
            raise SystemExit(1)
        return

    # Process videos
    extension = "json" if output_format == "json" else "md"
    schema = SEGMENTS_SCHEMA if is_segments_mode else None

    def analyze(video_input: str) -> VideoAnalysis:
        """Build the prompt for one input and dispatch it to the right backend."""
        kind = media_kind_for_input(video_input)

        # The LiteLLM backend has no Gemini Files API, so a files/... ref
        # cannot be resolved. Local files ARE supported: their bytes are sent
        # inline (base64) as OpenAI-compatible content parts.
        if isinstance(client, LiteLLMClient) and is_files_api_ref(video_input):
            raise click.ClickException(
                f"The LiteLLM backend cannot resolve Files API references; "
                f"{video_input!r} requires the Gemini Files API. Use the Gemini "
                f"API key (--api-key) or Vertex AI (--vertex) backend, or pass "
                f"the local file path directly (sent inline over LiteLLM)."
            )

        # Inject media duration into the prompt. Applied for all built-in modes
        # but skipped for custom --prompt, since user text may contain literal
        # braces that would break str.format().
        video_prompt = analysis_prompt
        if not prompt:
            template = analysis_prompt
            if kind == "audio":
                template = select_prompt(mode, "audio")
            video_prompt = template.format(
                duration_line=build_duration_line(
                    video_input,
                    clip_start=clip_start,
                    clip_end=clip_end,
                    verbose=verbose,
                )
                + offset_line
            )
        elif offset_line:
            video_prompt = f"{analysis_prompt}\n{offset_line}"

        common = {
            "fps": fps,
            "clip_start": clip_start,
            "clip_end": clip_end,
            "media_resolution": resolved_media_resolution,
            "thinking_level": thinking_level,
            "max_retries": max_retries,
        }

        if is_local_file(video_input):
            return process_local_file(
                client,
                video_input,
                video_prompt,
                model,
                verbose,
                schema,
                gcs_bucket=gcs_bucket,
                **common,
            )
        if is_files_api_ref(video_input):
            return process_files_api_ref(
                client, video_input, video_prompt, model, verbose, schema, **common
            )
        if is_gcs_uri(video_input):
            return process_gcs_uri(
                client, video_input, video_prompt, model, verbose, schema, **common
            )
        return process_video(
            client, video_input, video_prompt, model, schema, verbose=verbose, **common
        )

    def render(video_input: str, analysis: VideoAnalysis) -> str:
        """Format one result, running the segment split when requested."""
        kind = media_kind_for_input(video_input)

        if not (is_segments_mode and not analysis.error):
            if output_format == "json":
                return format_output_json(analysis, kind)
            return format_output_markdown(analysis, kind)

        # A malformed segments payload is this input's failure, not the run's:
        # record it so batch siblings still finish and the exit code reflects it.
        try:
            segments = parse_segments(analysis.raw_response)
        except click.ClickException as exc:
            analysis.error = str(exc)
            if output_format == "json":
                return format_output_json(analysis, kind)
            return format_output_markdown(analysis, kind)

        formatted = (
            format_segments_json(analysis, segments)
            if output_format == "json"
            else format_segments_markdown(analysis, segments)
        )

        if split and is_local_file(video_input):
            split_dir = output_dir_path or Path(video_input).resolve().parent
            click.echo(f"\nSplitting video into {len(segments)} segments...", err=True)
            created = split_video(video_input, segments, split_dir, verbose)
            click.echo(f"Created {len(created)} segment files in {split_dir}", err=True)
        elif split:
            click.echo(
                "Note: --split with segments mode on non-local files only "
                "outputs segment data. Use --split without --mode segments "
                "on YouTube URLs for chunked processing.",
                err=True,
            )
        return formatted

    def finish(video_input: str, analysis: VideoAnalysis) -> None:
        _handle_output(
            render(video_input, analysis),
            video_input,
            extension,
            output_dir_path,
            output_file,
            is_batch,
            verbose,
        )

    results: list[VideoAnalysis] = []

    def analyze_safely(video_input: str) -> VideoAnalysis:
        """Run one input, turning any escaping exception into a failed result.

        The process_* functions capture their own errors, but analyze() can
        raise before reaching them (backend guards, a prompt whose braces break
        .format(), an ffprobe timeout). Letting that propagate out of a batch
        would discard every sibling's output *after* the API calls had already
        been made and billed, since the executor drains before the exception
        surfaces. One bad input costs one input.
        """
        try:
            return analyze(video_input)
        except Exception as exc:
            return VideoAnalysis(
                url=video_input,
                processed_at=datetime.now().isoformat(),
                model=model,
                error=str(exc),
            )

    # Parallelize only when each result lands in its own file. Writing several
    # documents to stdout concurrently would interleave them into garbage.
    parallel = is_batch and workers > 1 and output_dir_path is not None

    if parallel:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        click.echo(
            f"Processing {len(inputs)} inputs with {workers} workers...", err=True
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(analyze_safely, item): item for item in inputs}
            for done in as_completed(futures):
                item = futures[done]
                analysis = done.result()
                results.append(analysis)
                status = "FAILED" if analysis.error else "ok"
                click.echo(
                    f"  [{len(results)}/{len(inputs)}] {item}: {status}", err=True
                )
                finish(item, analysis)
    else:
        with click.progressbar(
            inputs,
            label="Processing videos",
            show_pos=True,
            item_show_func=lambda x: x[:50] + "..." if x and len(x) > 50 else x,
        ) as progress_inputs:
            for video_input in progress_inputs:
                if verbose:
                    click.echo(f"\nProcessing: {video_input}")
                analysis = (
                    analyze_safely(video_input) if is_batch else analyze(video_input)
                )
                results.append(analysis)
                finish(video_input, analysis)

    # Summary for batch mode
    failed = [r for r in results if r.error]
    if is_batch:
        click.echo(
            f"\nProcessed {len(results)} videos: "
            f"{len(results) - len(failed)} successful, {len(failed)} failed"
        )
        if failed:
            click.echo("\nFailed videos:", err=True)
            for r in failed:
                click.echo(f"  - {r.url}: {r.error}", err=True)

    # Exit non-zero so callers and CI can actually detect failure. Writing an
    # error document and exiting 0 made every failure look like a success.
    if failed:
        raise SystemExit(1)
    if any(r.truncated for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
