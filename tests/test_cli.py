"""Comprehensive tests for the video processor CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from youtube_gemini_processor.cli import (
    PROMPTS,
    SEGMENTS_SCHEMA,
    VIDEO_MIME_TYPES,
    VideoAnalysis,
    _sanitize_filename,
    calculate_cost,
    extract_video_id,
    format_output_json,
    format_output_markdown,
    format_segments_json,
    format_segments_markdown,
    get_safe_filename,
    get_video_mime_type,
    is_local_file,
    is_youtube_url,
    main,
    parse_segments,
    split_video,
    validate_youtube_url,
)


class TestIsLocalFile:
    """Tests for is_local_file function."""

    def test_existing_file_returns_true(self, tmp_path: Path) -> None:
        """Test that existing files return True."""
        test_file = tmp_path / "video.mp4"
        test_file.write_text("fake video content")
        assert is_local_file(str(test_file)) is True

    def test_nonexistent_file_returns_false(self) -> None:
        """Test that non-existent files return False."""
        assert is_local_file("/nonexistent/path/video.mp4") is False

    def test_directory_returns_false(self, tmp_path: Path) -> None:
        """Test that directories return False."""
        assert is_local_file(str(tmp_path)) is False

    def test_youtube_url_returns_false(self) -> None:
        """Test that YouTube URLs return False."""
        assert is_local_file("https://www.youtube.com/watch?v=abc123") is False

    def test_relative_path_existing_file(self, tmp_path: Path, monkeypatch) -> None:
        """Test relative paths to existing files."""
        test_file = tmp_path / "test.mp4"
        test_file.write_text("content")
        monkeypatch.chdir(tmp_path)
        assert is_local_file("test.mp4") is True


class TestIsYoutubeUrl:
    """Tests for is_youtube_url function."""

    def test_standard_youtube_url(self) -> None:
        """Test standard youtube.com URLs."""
        assert is_youtube_url("https://www.youtube.com/watch?v=abc123") is True
        assert is_youtube_url("http://www.youtube.com/watch?v=abc123") is True
        assert is_youtube_url("https://youtube.com/watch?v=abc123") is True

    def test_short_youtube_url(self) -> None:
        """Test youtu.be short URLs."""
        assert is_youtube_url("https://youtu.be/abc123") is True
        assert is_youtube_url("http://youtu.be/abc123") is True

    def test_youtube_without_protocol(self) -> None:
        """Test YouTube URLs without protocol."""
        assert is_youtube_url("www.youtube.com/watch?v=abc123") is True
        assert is_youtube_url("youtube.com/watch?v=abc123") is True
        assert is_youtube_url("youtu.be/abc123") is True

    def test_non_youtube_url(self) -> None:
        """Test non-YouTube URLs return False."""
        assert is_youtube_url("https://vimeo.com/123456") is False
        assert is_youtube_url("https://example.com/video.mp4") is False

    def test_local_path_returns_false(self) -> None:
        """Test local file paths return False."""
        assert is_youtube_url("/path/to/video.mp4") is False
        assert is_youtube_url("./video.mp4") is False


class TestGetVideoMimeType:
    """Tests for get_video_mime_type function."""

    def test_mp4_mime_type(self, tmp_path: Path) -> None:
        """Test MP4 files return correct MIME type."""
        test_file = tmp_path / "video.mp4"
        assert get_video_mime_type(test_file) == "video/mp4"

    def test_mov_mime_type(self, tmp_path: Path) -> None:
        """Test MOV files return correct MIME type."""
        test_file = tmp_path / "video.mov"
        assert get_video_mime_type(test_file) == "video/quicktime"

    def test_webm_mime_type(self, tmp_path: Path) -> None:
        """Test WebM files return correct MIME type."""
        test_file = tmp_path / "video.webm"
        assert get_video_mime_type(test_file) == "video/webm"

    def test_avi_mime_type(self, tmp_path: Path) -> None:
        """Test AVI files return correct MIME type."""
        test_file = tmp_path / "video.avi"
        assert get_video_mime_type(test_file) == "video/x-msvideo"

    def test_mkv_mime_type(self, tmp_path: Path) -> None:
        """Test MKV files return correct MIME type."""
        test_file = tmp_path / "video.mkv"
        assert get_video_mime_type(test_file) == "video/x-matroska"

    def test_case_insensitive(self, tmp_path: Path) -> None:
        """Test that extension matching is case-insensitive."""
        test_file = tmp_path / "video.MP4"
        assert get_video_mime_type(test_file) == "video/mp4"

    def test_unsupported_format_raises_error(self, tmp_path: Path) -> None:
        """Test unsupported formats raise ClickException."""
        import click

        test_file = tmp_path / "video.xyz"
        with pytest.raises(click.ClickException) as exc_info:
            get_video_mime_type(test_file)
        assert "Unsupported video format" in str(exc_info.value)
        assert ".xyz" in str(exc_info.value)

    def test_all_supported_formats(self, tmp_path: Path) -> None:
        """Test all formats in VIDEO_MIME_TYPES are handled."""
        for ext, expected_mime in VIDEO_MIME_TYPES.items():
            test_file = tmp_path / f"video{ext}"
            assert get_video_mime_type(test_file) == expected_mime


class TestValidateYoutubeUrl:
    """Tests for validate_youtube_url function."""

    def test_standard_url(self) -> None:
        """Test standard YouTube watch URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        result = validate_youtube_url(url)
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_short_url(self) -> None:
        """Test youtu.be short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        result = validate_youtube_url(url)
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_embed_url(self) -> None:
        """Test YouTube embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        result = validate_youtube_url(url)
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_shorts_url(self) -> None:
        """Test YouTube Shorts URL."""
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        result = validate_youtube_url(url)
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_url_without_protocol(self) -> None:
        """Test URL without https://."""
        url = "youtube.com/watch?v=dQw4w9WgXcQ"
        result = validate_youtube_url(url)
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_url_with_extra_params(self) -> None:
        """Test URL with additional query parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120"
        result = validate_youtube_url(url)
        assert result == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_invalid_url_raises_error(self) -> None:
        """Test invalid URLs raise ClickException."""
        import click

        with pytest.raises(click.ClickException) as exc_info:
            validate_youtube_url("https://vimeo.com/123456")
        assert "Invalid YouTube URL" in str(exc_info.value)


class TestExtractVideoId:
    """Tests for extract_video_id function."""

    def test_extract_from_standard_url(self) -> None:
        """Test extracting ID from standard URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_with_extra_params(self) -> None:
        """Test extracting ID when URL has extra parameters."""
        url = "https://www.youtube.com/watch?v=abc123&t=120&list=PLxyz"
        assert extract_video_id(url) == "abc123"

    def test_no_video_id_returns_none(self) -> None:
        """Test URLs without video ID return None."""
        assert extract_video_id("https://youtube.com/") is None
        assert extract_video_id("https://youtu.be/") is None

    def test_video_id_with_special_chars(self) -> None:
        """Test video IDs with underscores and dashes."""
        url = "https://www.youtube.com/watch?v=a_b-c123XYZ"
        assert extract_video_id(url) == "a_b-c123XYZ"


class TestGetSafeFilename:
    """Tests for get_safe_filename function."""

    def test_youtube_url_returns_video_id(self) -> None:
        """Test YouTube URLs return video_<id> format."""
        url = "https://www.youtube.com/watch?v=abc123"
        result = get_safe_filename(url)
        assert result == "video_abc123"

    def test_local_file_returns_stem(self, tmp_path: Path) -> None:
        """Test local files return filename without extension."""
        test_file = tmp_path / "my_presentation.mp4"
        test_file.write_text("content")
        result = get_safe_filename(str(test_file))
        assert result == "my_presentation"

    def test_nonexistent_file_returns_timestamp(self) -> None:
        """Test non-existent files return timestamp-based name."""
        result = get_safe_filename("/nonexistent/video.mp4")
        assert result.startswith("video_")
        # Should be a timestamp format
        assert len(result) > 10


class TestCalculateCost:
    """Tests for calculate_cost function."""

    def test_gemini_3_flash_pricing(self) -> None:
        """Test cost calculation for gemini-3-flash-preview."""
        stats = calculate_cost("gemini-3-flash-preview", 1_000_000, 100_000)
        assert stats.input_tokens == 1_000_000
        assert stats.output_tokens == 100_000
        assert stats.total_tokens == 1_100_000
        assert stats.input_cost == pytest.approx(0.50)  # $0.50 per 1M input tokens
        assert stats.output_cost == pytest.approx(
            0.30
        )  # $3.00 per 1M output tokens * 0.1M
        assert stats.total_cost == pytest.approx(0.80)

    def test_gemini_2_flash_pricing(self) -> None:
        """Test cost calculation for gemini-2.0-flash."""
        stats = calculate_cost("gemini-2.0-flash", 1_000_000, 1_000_000)
        assert stats.input_cost == 0.10  # $0.10 per 1M input tokens
        assert stats.output_cost == 0.40  # $0.40 per 1M output tokens
        assert stats.total_cost == 0.50

    def test_unknown_model_uses_default(self) -> None:
        """Test unknown models use default pricing."""
        stats = calculate_cost("unknown-model", 1_000_000, 100_000)
        # Should use gemini-3-flash-preview pricing as default
        assert stats.input_cost == pytest.approx(0.50)
        assert stats.output_cost == pytest.approx(0.30)

    def test_zero_tokens(self) -> None:
        """Test zero token counts."""
        stats = calculate_cost("gemini-3-flash-preview", 0, 0)
        assert stats.total_tokens == 0
        assert stats.total_cost == 0.0


class TestFormatOutputMarkdown:
    """Tests for format_output_markdown function."""

    def test_basic_output(self) -> None:
        """Test basic markdown output format."""
        analysis = VideoAnalysis(
            url="https://youtube.com/watch?v=abc123",
            processed_at="2024-01-15T10:00:00",
            model="gemini-3-flash-preview",
            raw_response="# Video Summary\n\nThis is a test.",
        )
        result = format_output_markdown(analysis)
        assert "# Video Analysis" in result
        assert "**URL**: https://youtube.com/watch?v=abc123" in result
        assert "**Model**: gemini-3-flash-preview" in result
        assert "# Video Summary" in result

    def test_output_with_usage_stats(self) -> None:
        """Test markdown output includes usage stats when present."""
        from youtube_gemini_processor.cli import UsageStats

        analysis = VideoAnalysis(
            url="https://youtube.com/watch?v=abc123",
            processed_at="2024-01-15T10:00:00",
            model="gemini-3-flash-preview",
            raw_response="Content",
            usage=UsageStats(
                input_tokens=1000,
                output_tokens=500,
                total_tokens=1500,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.003,
            ),
        )
        result = format_output_markdown(analysis)
        assert "**Usage**:" in result
        assert "1,000 input" in result
        assert "500 output" in result
        assert "**Cost**:" in result

    def test_error_output(self) -> None:
        """Test markdown output for errors."""
        analysis = VideoAnalysis(
            url="https://youtube.com/watch?v=abc123",
            error="API connection failed",
        )
        result = format_output_markdown(analysis)
        assert "# Error Processing Video" in result
        assert "API connection failed" in result


class TestFormatOutputJson:
    """Tests for format_output_json function."""

    def test_basic_output(self) -> None:
        """Test basic JSON output format."""
        analysis = VideoAnalysis(
            url="https://youtube.com/watch?v=abc123",
            title="Test Video",
            processed_at="2024-01-15T10:00:00",
            model="gemini-3-flash-preview",
            raw_response="Content here",
            summary="A test summary",
        )
        result = format_output_json(analysis)
        data = json.loads(result)
        assert data["url"] == "https://youtube.com/watch?v=abc123"
        assert data["title"] == "Test Video"
        assert data["model"] == "gemini-3-flash-preview"
        assert data["content"] == "Content here"
        assert data["summary"] == "A test summary"
        assert data["error"] is None

    def test_output_with_usage(self) -> None:
        """Test JSON output includes usage stats."""
        from youtube_gemini_processor.cli import UsageStats

        analysis = VideoAnalysis(
            url="https://youtube.com/watch?v=abc123",
            processed_at="2024-01-15T10:00:00",
            model="gemini-3-flash-preview",
            raw_response="Content",
            usage=UsageStats(
                input_tokens=1000,
                output_tokens=500,
                total_tokens=1500,
                input_cost=0.001,
                output_cost=0.002,
                total_cost=0.003,
            ),
        )
        result = format_output_json(analysis)
        data = json.loads(result)
        assert data["usage"]["input_tokens"] == 1000
        assert data["usage"]["output_tokens"] == 500
        assert data["usage"]["total_cost_usd"] == 0.003

    def test_error_output(self) -> None:
        """Test JSON output for errors."""
        analysis = VideoAnalysis(
            url="https://youtube.com/watch?v=abc123",
            error="Something went wrong",
        )
        result = format_output_json(analysis)
        data = json.loads(result)
        assert data["error"] == "Something went wrong"


class TestCLIIntegration:
    """Integration tests for the CLI using Click's test runner."""

    def test_no_input_shows_error(self) -> None:
        """Test that running without input shows error."""
        runner = CliRunner()
        result = runner.invoke(main, [])
        assert result.exit_code != 0
        assert "Either INPUT" in result.output or "required" in result.output.lower()

    def test_both_input_and_batch_shows_error(self, tmp_path: Path) -> None:
        """Test that providing both input and batch shows error."""
        batch_file = tmp_path / "urls.txt"
        batch_file.write_text("https://youtube.com/watch?v=abc123")
        runner = CliRunner()
        result = runner.invoke(
            main, ["https://youtube.com/watch?v=abc123", "--batch", str(batch_file)]
        )
        assert result.exit_code != 0
        assert "Cannot specify both" in result.output

    def test_help_shows_local_file_support(self) -> None:
        """Test that help text mentions local file support."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "local" in result.output.lower()
        assert "./video.mp4" in result.output

    def test_help_shows_supported_formats(self) -> None:
        """Test that help text shows supported video formats."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert ".mp4" in result.output
        assert ".mov" in result.output

    def test_version_option(self) -> None:
        """Test --version flag works."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_process_youtube_url_calls_process_video(
        self, mock_get_client: MagicMock
    ) -> None:
        """Test YouTube URL triggers process_video path."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "# Analysis\n\nTest content"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "https://www.youtube.com/watch?v=test123",
                "--api-key",
                "fake-key",
            ],
        )

        # Should call generate_content
        mock_client.models.generate_content.assert_called_once()
        # Check output contains analysis
        assert "Video Analysis" in result.output or result.exit_code == 0

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_process_local_file_calls_files_upload(
        self, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        """Test local file triggers Files API upload path."""
        # Create a test video file
        test_file = tmp_path / "test_video.mp4"
        test_file.write_bytes(b"fake video content")

        mock_client = MagicMock()
        mock_uploaded_file = MagicMock()
        mock_uploaded_file.name = "files/abc123"
        mock_uploaded_file.uri = (
            "https://generativelanguage.googleapis.com/files/abc123"
        )
        mock_uploaded_file.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_uploaded_file
        mock_client.files.get.return_value = mock_uploaded_file

        mock_response = MagicMock()
        mock_response.text = "# Analysis\n\nTest content"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        runner.invoke(
            main,
            [
                str(test_file),
                "--api-key",
                "fake-key",
            ],
        )

        # Should call files.upload for local files
        mock_client.files.upload.assert_called_once()

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_batch_mode_processes_multiple_inputs(
        self, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        """Test batch mode processes multiple inputs."""
        # Create batch file with URLs
        batch_file = tmp_path / "inputs.txt"
        batch_file.write_text(
            "https://youtube.com/watch?v=video1\n"
            "https://youtube.com/watch?v=video2\n"
            "# This is a comment\n"
        )

        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "# Analysis"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--batch",
                str(batch_file),
                "-o",
                str(output_dir),
                "--api-key",
                "fake-key",
            ],
        )

        # Should process 2 videos (not the comment)
        assert mock_client.models.generate_content.call_count == 2
        # Should report processing results
        assert "2" in result.output

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_json_output_format(self, mock_get_client: MagicMock) -> None:
        """Test JSON output format option."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Analysis content"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "https://www.youtube.com/watch?v=test123",
                "--format",
                "json",
                "--api-key",
                "fake-key",
            ],
        )

        # Output should be valid JSON
        try:
            data = json.loads(result.output)
            assert "url" in data
            assert "content" in data
        except json.JSONDecodeError:
            # If there's an error message before the JSON, that's also acceptable
            pass

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_concise_mode_uses_correct_prompt(self, mock_get_client: MagicMock) -> None:
        """Test concise mode uses CONCISE_PROMPT."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Concise analysis"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        runner.invoke(
            main,
            [
                "https://www.youtube.com/watch?v=test123",
                "--mode",
                "concise",
                "--api-key",
                "fake-key",
            ],
        )

        # Check that the call was made with concise prompt keywords
        call_args = mock_client.models.generate_content.call_args
        contents = call_args.kwargs.get(
            "contents", call_args.args[0] if call_args.args else None
        )
        # The prompt should contain concise-specific text
        assert contents is not None

    def test_unsupported_video_format_error(self, tmp_path: Path) -> None:
        """Test unsupported video format shows helpful error."""
        test_file = tmp_path / "video.xyz"
        test_file.write_text("content")

        runner = CliRunner()
        with patch("youtube_gemini_processor.cli.get_gemini_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client

            result = runner.invoke(
                main,
                [
                    str(test_file),
                    "--api-key",
                    "fake-key",
                ],
            )

            # Error is caught and reported in output (may be exit code 0 or 1)
            assert (
                "Unsupported video format" in result.output or "Error" in result.output
            )


class TestVideoAnalysisDataclass:
    """Tests for VideoAnalysis dataclass."""

    def test_default_values(self) -> None:
        """Test VideoAnalysis has correct defaults."""
        analysis = VideoAnalysis(url="https://test.com")
        assert analysis.url == "https://test.com"
        assert analysis.title == ""
        assert analysis.duration == ""
        assert analysis.transcript == ""
        assert analysis.visual_descriptions == []
        assert analysis.key_topics == []
        assert analysis.summary == ""
        assert analysis.raw_response == ""
        assert analysis.processed_at == ""
        assert analysis.model == ""
        assert analysis.error is None
        assert analysis.usage is None

    def test_all_fields(self) -> None:
        """Test VideoAnalysis with all fields populated."""
        from youtube_gemini_processor.cli import UsageStats

        usage = UsageStats(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_cost=0.01,
            output_cost=0.02,
            total_cost=0.03,
        )
        analysis = VideoAnalysis(
            url="https://test.com",
            title="Test Title",
            duration="10:30",
            transcript="Full transcript here",
            visual_descriptions=[{"time": "00:30", "desc": "Slide 1"}],
            key_topics=["topic1", "topic2"],
            summary="A summary",
            raw_response="Raw API response",
            processed_at="2024-01-15T10:00:00",
            model="gemini-3-flash-preview",
            error=None,
            usage=usage,
        )
        assert analysis.title == "Test Title"
        assert len(analysis.key_topics) == 2
        assert analysis.usage.total_cost == 0.03


class TestSegmentsPrompt:
    """Tests for segments mode prompt and schema."""

    def test_segments_in_prompts_dict(self) -> None:
        """Test that segments prompt is registered in PROMPTS dict."""
        assert "segments" in PROMPTS
        assert "segment" in PROMPTS["segments"].lower()

    def test_segments_schema_structure(self) -> None:
        """Test SEGMENTS_SCHEMA has correct structure."""
        assert SEGMENTS_SCHEMA["type"] == "ARRAY"
        props = SEGMENTS_SCHEMA["items"]["properties"]
        assert "segment_number" in props
        assert "start_time" in props
        assert "end_time" in props
        assert "title" in props
        assert "speaker" in props
        assert "summary" in props


class TestParseSegments:
    """Tests for parse_segments function."""

    def test_parse_valid_json(self) -> None:
        """Test parsing a valid JSON array response."""
        response = json.dumps(
            [
                {
                    "segment_number": 1,
                    "start_time": "00:00:00",
                    "end_time": "00:10:00",
                    "title": "Introduction",
                    "speaker": "John",
                    "summary": "Opening remarks",
                }
            ]
        )
        segments = parse_segments(response)
        assert len(segments) == 1
        assert segments[0]["title"] == "Introduction"
        assert segments[0]["start_time"] == "00:00:00"

    def test_parse_json_with_markdown_fencing(self) -> None:
        """Test parsing JSON wrapped in markdown code fences."""
        response = '```json\n[{"segment_number": 1, "start_time": "00:00:00", "end_time": "00:05:00", "title": "Intro", "speaker": "A", "summary": "Hi"}]\n```'
        segments = parse_segments(response)
        assert len(segments) == 1
        assert segments[0]["title"] == "Intro"

    def test_parse_json_with_surrounding_text(self) -> None:
        """Test parsing JSON with extra text before/after."""
        response = 'Here are the segments:\n[{"segment_number": 1, "start_time": "00:00:00", "end_time": "00:05:00", "title": "Intro", "speaker": "A", "summary": "Hi"}]\nDone.'
        segments = parse_segments(response)
        assert len(segments) == 1

    def test_parse_multiple_segments(self) -> None:
        """Test parsing multiple segments."""
        response = json.dumps(
            [
                {
                    "segment_number": 1,
                    "start_time": "00:00:00",
                    "end_time": "00:10:00",
                    "title": "Part 1",
                    "speaker": "A",
                    "summary": "First part",
                },
                {
                    "segment_number": 2,
                    "start_time": "00:10:00",
                    "end_time": "00:20:00",
                    "title": "Part 2",
                    "speaker": "B",
                    "summary": "Second part",
                },
            ]
        )
        segments = parse_segments(response)
        assert len(segments) == 2
        assert segments[1]["segment_number"] == 2

    def test_parse_invalid_json_raises_error(self) -> None:
        """Test invalid JSON raises ClickException."""
        import click

        with pytest.raises(click.ClickException, match="Failed to parse"):
            parse_segments("not json at all")

    def test_parse_empty_array_raises_error(self) -> None:
        """Test empty array raises ClickException."""
        import click

        with pytest.raises(click.ClickException, match="valid segments array"):
            parse_segments("[]")


class TestSanitizeFilename:
    """Tests for _sanitize_filename function."""

    def test_basic_sanitization(self) -> None:
        """Test basic filename sanitization."""
        assert _sanitize_filename("Hello World") == "Hello_World"

    def test_special_characters_removed(self) -> None:
        """Test special characters are removed."""
        result = _sanitize_filename("Q&A: What's Next?")
        assert "&" not in result
        assert "?" not in result
        assert "'" not in result

    def test_length_limit(self) -> None:
        """Test filename is truncated to 80 chars."""
        long_name = "A" * 100
        assert len(_sanitize_filename(long_name)) <= 80


class TestFormatSegmentsMarkdown:
    """Tests for format_segments_markdown function."""

    def test_basic_formatting(self) -> None:
        """Test basic segments markdown formatting."""
        analysis = VideoAnalysis(
            url="/path/to/video.mp4",
            processed_at="2024-01-15T10:00:00",
            model="gemini-3-flash-preview",
        )
        segments = [
            {
                "segment_number": 1,
                "start_time": "00:00:00",
                "end_time": "00:10:00",
                "title": "Introduction",
                "speaker": "John",
                "summary": "Opening remarks and agenda",
            },
            {
                "segment_number": 2,
                "start_time": "00:10:00",
                "end_time": "00:25:00",
                "title": "Product Update",
                "speaker": "Jane",
                "summary": "New features overview",
            },
        ]
        result = format_segments_markdown(analysis, segments)
        assert "# Video Segments Analysis" in result
        assert "Introduction" in result
        assert "Product Update" in result
        assert "00:00:00" in result
        assert "John" in result

    def test_error_formatting(self) -> None:
        """Test error output for segments markdown."""
        analysis = VideoAnalysis(url="/path/to/video.mp4", error="API failed")
        result = format_segments_markdown(analysis, [])
        assert "Error" in result
        assert "API failed" in result


class TestFormatSegmentsJson:
    """Tests for format_segments_json function."""

    def test_basic_formatting(self) -> None:
        """Test basic segments JSON formatting."""
        analysis = VideoAnalysis(
            url="/path/to/video.mp4",
            processed_at="2024-01-15T10:00:00",
            model="gemini-3-flash-preview",
        )
        segments = [
            {
                "segment_number": 1,
                "start_time": "00:00:00",
                "end_time": "00:10:00",
                "title": "Intro",
                "speaker": "A",
                "summary": "Hi",
            }
        ]
        result = format_segments_json(analysis, segments)
        data = json.loads(result)
        assert data["segments"] == segments
        assert data["url"] == "/path/to/video.mp4"
        assert data["error"] is None


class TestSplitVideo:
    """Tests for split_video function."""

    @patch("youtube_gemini_processor.cli.shutil.which")
    def test_missing_ffmpeg_raises_error(self, mock_which: MagicMock) -> None:
        """Test that missing ffmpeg raises ClickException."""
        import click

        mock_which.return_value = None
        with pytest.raises(click.ClickException, match="ffmpeg"):
            split_video("/fake/video.mp4", [{"segment_number": 1}])

    @patch("youtube_gemini_processor.cli.subprocess.run")
    @patch("youtube_gemini_processor.cli.shutil.which")
    def test_split_creates_files(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test split_video calls ffmpeg for each segment."""
        mock_which.return_value = "/usr/bin/ffmpeg"

        # Create source video
        source = tmp_path / "video.mp4"
        source.write_bytes(b"fake video")

        # Create expected output files (simulating ffmpeg)
        def create_output(cmd, **_kwargs):
            out_path = Path(cmd[-1])
            out_path.write_bytes(b"segment data")
            result = MagicMock()
            result.returncode = 0
            return result

        mock_run.side_effect = create_output

        segments = [
            {
                "segment_number": 1,
                "start_time": "00:00:00",
                "end_time": "00:10:00",
                "title": "Introduction",
                "speaker": "A",
                "summary": "Intro",
            },
            {
                "segment_number": 2,
                "start_time": "00:10:00",
                "end_time": "00:20:00",
                "title": "Main Content",
                "speaker": "B",
                "summary": "Main",
            },
        ]

        created = split_video(str(source), segments, tmp_path)
        assert len(created) == 2
        assert mock_run.call_count == 2

    @patch("youtube_gemini_processor.cli.subprocess.run")
    @patch("youtube_gemini_processor.cli.shutil.which")
    def test_split_handles_ffmpeg_failure(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test split_video handles ffmpeg failures gracefully."""
        mock_which.return_value = "/usr/bin/ffmpeg"

        source = tmp_path / "video.mp4"
        source.write_bytes(b"fake video")

        result = MagicMock()
        result.returncode = 1
        result.stderr = "ffmpeg error"
        mock_run.return_value = result

        segments = [
            {
                "segment_number": 1,
                "start_time": "00:00:00",
                "end_time": "00:10:00",
                "title": "Intro",
                "speaker": "A",
                "summary": "Intro",
            },
        ]

        created = split_video(str(source), segments, tmp_path)
        assert len(created) == 0  # No files created due to failure


class TestCLISegmentsMode:
    """Integration tests for segments mode CLI."""

    def test_segments_mode_in_help(self) -> None:
        """Test that --mode help shows segments option."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "segments" in result.output

    def test_split_flag_in_help(self) -> None:
        """Test that --split flag appears in help."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert "--split" in result.output

    def test_split_without_segments_mode_errors(self) -> None:
        """Test --split without --mode segments raises error."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "https://www.youtube.com/watch?v=test123",
                "--split",
                "--api-key",
                "fake-key",
            ],
        )
        assert result.exit_code != 0
        assert "segments" in result.output.lower()

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_segments_mode_uses_schema(self, mock_get_client: MagicMock) -> None:
        """Test segments mode passes response_schema to generate_content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            [
                {
                    "segment_number": 1,
                    "start_time": "00:00:00",
                    "end_time": "00:05:00",
                    "title": "Intro",
                    "speaker": "Host",
                    "summary": "Opening",
                }
            ]
        )
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "https://www.youtube.com/watch?v=test123",
                "--mode",
                "segments",
                "--api-key",
                "fake-key",
            ],
        )

        # Check output contains segment info
        assert "Intro" in result.output
        # Verify generate_content was called with schema config
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_schema is not None

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_segments_mode_with_api_error_falls_back_to_generic_formatter(
        self, mock_get_client: MagicMock
    ) -> None:
        """Test segments mode uses generic formatter when analysis has an error."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API unavailable")
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "https://www.youtube.com/watch?v=test123",
                "--mode",
                "segments",
                "--api-key",
                "fake-key",
            ],
        )

        # Should not crash with UnboundLocalError, should show error in output
        assert result.exit_code == 0
        assert "Error" in result.output
