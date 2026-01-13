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


# Default comprehensive analysis prompt
DEFAULT_PROMPT = """Analyze this YouTube video comprehensively. Provide:

## 1. VIDEO METADATA
- Title of the video
- Estimated duration
- Speaker(s)/Presenter(s) if identifiable

## 2. FULL TRANSCRIPT
Provide a complete transcript of all spoken content with timestamps in [MM:SS] format.
Format as:
[00:00] Speaker: "Spoken text..."
[00:15] Speaker: "Next part..."

## 3. VISUAL CONTENT DESCRIPTIONS
For each significant visual element (slides, diagrams, charts, demonstrations, screen shares), provide:
- [MM:SS] Description of what is shown visually
- Any text visible on slides/screens
- Diagrams or charts explained

Format as:
### Visual at [MM:SS]
**Type**: Slide/Diagram/Chart/Demo/Screen
**Content**: Detailed description of visual content
**Text on screen**: Any readable text

## 4. KEY TOPICS & TIMESTAMPS
List main topics covered with their timestamps:
- [MM:SS] Topic 1
- [MM:SS] Topic 2

## 5. SUMMARY
A comprehensive summary of the video content (300-500 words).

## 6. KEY TAKEAWAYS
Bullet points of the most important insights from this video.

Be thorough and detailed. Include ALL visual content descriptions, especially slides and presentations."""


CONCISE_PROMPT = """Analyze this YouTube video. Provide:

1. **Title & Duration**
2. **Summary** (2-3 paragraphs)
3. **Key Topics** with timestamps [MM:SS]
4. **Main Takeaways** (bullet points)
5. **Notable Visuals** - Describe any important slides, diagrams, or demonstrations shown

Be concise but comprehensive."""


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


def get_gemini_client(api_key: str | None = None):
    """Initialize Gemini client with API key."""
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise click.ClickException(
            "Gemini API key required. Set GEMINI_API_KEY or GOOGLE_API_KEY "
            "environment variable, or pass --api-key"
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
    model: str = "gemini-2.5-flash",
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
                    parts=[
                        types.Part(
                            file_data=types.FileData(file_uri=normalized_url)
                        ),
                        types.Part(text=prompt),
                    ]
                )
            ],
        )

        analysis.raw_response = response.text

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

    output = f"""# Video Analysis

**URL**: {analysis.url}
**Processed**: {analysis.processed_at}
**Model**: {analysis.model}

---

{analysis.raw_response}
"""
    return output


def format_output_json(analysis: VideoAnalysis) -> str:
    """Format analysis as JSON."""
    return json.dumps(
        {
            "url": analysis.url,
            "title": analysis.title,
            "processed_at": analysis.processed_at,
            "model": analysis.model,
            "content": analysis.raw_response,
            "summary": analysis.summary,
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
    type=str,
    default="gemini-2.5-flash",
    help="Gemini model to use (default: gemini-2.5-flash)",
)
@click.option(
    "--api-key",
    envvar="GEMINI_API_KEY",
    help="Gemini API key (or set GEMINI_API_KEY env var)",
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
    Environment Variables:
        GEMINI_API_KEY    Google Gemini API key
        GOOGLE_API_KEY    Alternative API key variable
    """
    if not url and not batch:
        raise click.ClickException("Either URL or --batch file is required")

    if url and batch:
        raise click.ClickException("Cannot specify both URL and --batch")

    # Initialize client
    client = get_gemini_client(api_key)

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
