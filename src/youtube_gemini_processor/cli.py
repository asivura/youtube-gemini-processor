#!/usr/bin/env python3
"""
YouTube Video Processor CLI

A universal CLI tool to process any YouTube video using Google's Gemini API.
Extracts comprehensive content including transcripts and visual descriptions.

Usage:
    yt-process <youtube_url> [options]
    yt-process --batch <file_with_urls> [options]

Examples:
    # Process a single video
    yt-process "https://www.youtube.com/watch?v=VIDEO_ID"

    # Process with custom output
    yt-process "https://youtube.com/watch?v=XYZ" -o output.md

    # Process multiple videos from a file
    yt-process --batch urls.txt -o ./output_dir/

    # Get JSON output
    yt-process "https://youtube.com/watch?v=XYZ" --format json

    # Custom analysis prompt
    yt-process "https://youtube.com/watch?v=XYZ" --prompt "Focus on technical details"
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from typing import Literal


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
DEFAULT_PROMPT = """You are an expert content analyst. Analyze this YouTube video with extreme thoroughness and produce a comprehensive markdown document.

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

*Analysis generated by YouTube Gemini Processor*

# CRITICAL REMINDERS
- Include EVERY slide and visual - do not skip any
- Transcribe ALL text shown on screen
- Capture the FULL transcript, not a summary
- Include DETAILED SPEAKER NOTES for each slide - what the speaker explains, examples they give, stories they tell
- Capture content spoken BEFORE the first slide and BETWEEN slides
- Be extremely thorough - more detail is better
- Use proper markdown tables, headers, and formatting"""


CONCISE_PROMPT = """Analyze this YouTube video. Provide:

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


TRANSCRIPT_ONLY_PROMPT = """Provide a complete transcript of this YouTube video.

Format with timestamps:
[MM:SS] "Spoken text..."

Include speaker identification if multiple speakers.
Also note any significant visual content shown (slides, demos) in brackets like:
[MM:SS] [SLIDE: Title of slide or description]
[MM:SS] [DEMO: What is being demonstrated]"""


PROMPTS = {
    "comprehensive": DEFAULT_PROMPT,
    "concise": CONCISE_PROMPT,
    "transcript": TRANSCRIPT_ONLY_PROMPT,
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


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> UsageStats:
    """Calculate usage cost based on model and token counts."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gemini-3-flash-preview"])

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

        click.echo(f"Using Vertex AI (project: {gcp_project}, location: {gcp_location})", err=True)

        return genai.Client(
            vertexai=True,
            project=gcp_project,
            location=gcp_location,
        )

    # API key authentication
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
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


def process_video(
    client,
    url: str,
    prompt: str,
    model: str = "gemini-3-flash-preview",
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
                    ]
                )
            ],
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


def get_safe_filename(url: str) -> str:
    """Generate safe filename from YouTube URL."""
    video_id = extract_video_id(url)
    if video_id:
        return f"video_{video_id}"
    return f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


@click.command()
@click.argument("url", required=False)
@click.option(
    "--batch",
    "-b",
    type=click.Path(exists=True),
    help="File containing YouTube URLs (one per line)",
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
    type=click.Choice(["comprehensive", "concise", "transcript"]),
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
    type=click.Choice([
        "gemini-3-flash-preview",
        "gemini-3-pro-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    ]),
    default="gemini-3-flash-preview",
    help="Gemini model to use (default: gemini-3-flash-preview)",
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
@click.version_option()
def main(
    url: str | None,
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
):
    """
    Process YouTube videos using Google's Gemini API.

    Extracts comprehensive content including transcripts and visual descriptions
    (slides, diagrams, charts, demonstrations).

    \b
    Examples:
        # Process a single video
        yt-process "https://www.youtube.com/watch?v=VIDEO_ID"

        # Save to file
        yt-process "https://youtube.com/watch?v=XYZ" -o analysis.md

        # Quick summary mode
        yt-process "https://youtube.com/watch?v=XYZ" -m concise

        # Transcript only
        yt-process "https://youtube.com/watch?v=XYZ" -m transcript

        # Batch process from file
        yt-process --batch urls.txt -o ./output/

        # JSON output
        yt-process "https://youtube.com/watch?v=XYZ" -f json

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
    if not url and not batch:
        raise click.ClickException("Either URL or --batch file is required")

    if url and batch:
        raise click.ClickException("Cannot specify both URL and --batch")

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
    analysis_prompt = prompt if prompt else PROMPTS[mode]

    # Collect URLs to process
    urls: list[str] = []
    if batch:
        batch_path = Path(batch)
        urls = [
            line.strip()
            for line in batch_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        if verbose:
            click.echo(f"Found {len(urls)} URLs in batch file")
    else:
        urls = [url]  # type: ignore

    # Determine output handling
    is_batch = len(urls) > 1
    output_dir: Path | None = None
    output_file: Path | None = None

    if output:
        output_path = Path(output)
        if is_batch or output_path.is_dir():
            output_dir = output_path
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_file = output_path

    # Process videos
    results: list[VideoAnalysis] = []
    formatter = format_output_json if output_format == "json" else format_output_markdown
    extension = "json" if output_format == "json" else "md"

    with click.progressbar(
        urls,
        label="Processing videos",
        show_pos=True,
        item_show_func=lambda x: x[:50] + "..." if x and len(x) > 50 else x,
    ) as progress_urls:
        for video_url in progress_urls:
            if verbose:
                click.echo(f"\nProcessing: {video_url}")

            analysis = process_video(client, video_url, analysis_prompt, model)
            results.append(analysis)

            # Output handling
            formatted = formatter(analysis)

            if output_dir:
                # Batch mode: write each to separate file
                filename = f"{get_safe_filename(video_url)}.{extension}"
                file_path = output_dir / filename
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
        click.echo(f"\nProcessed {len(results)} videos: {successful} successful, {failed} failed")

        if failed > 0 and verbose:
            click.echo("\nFailed videos:")
            for r in results:
                if r.error:
                    click.echo(f"  - {r.url}: {r.error}")


if __name__ == "__main__":
    main()
