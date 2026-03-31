"""Additional tests to boost coverage to 95%+.

Covers: get_gemini_client, process_video, process_local_file, process_gcs_uri,
_call_gemini_and_parse, fetch_youtube_chapters, get_video_duration,
get_video_duration_gcs, split_youtube_video, _process_single_chapter,
_handle_file_management, _handle_upload_only, _handle_chapter_splitting,
_handle_output, _format_duration, _normalize_timestamp_to_hhmmss,
and various main() branches.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from youtube_gemini_processor.cli import (
    VideoAnalysis,
    _format_duration,
    _handle_output,
    _normalize_timestamp_to_hhmmss,
    _process_single_chapter,
    fetch_youtube_chapters,
    format_output_markdown,
    format_segments_json,
    format_segments_markdown,
    get_gemini_client,
    get_video_duration,
    get_video_duration_gcs,
    main,
    process_gcs_uri,
    process_local_file,
    process_video,
    split_video,
    split_youtube_video,
)


# ---------------------------------------------------------------------------
# get_gemini_client
# ---------------------------------------------------------------------------
class TestGetGeminiClient:
    """Tests for get_gemini_client authentication logic."""

    @patch("google.genai.Client")
    @patch.dict("os.environ", {"GOOGLE_GENAI_USE_VERTEXAI": ""})
    def test_api_key_auth(self, mock_client_cls: MagicMock) -> None:
        get_gemini_client(api_key="test-key")
        mock_client_cls.assert_called_once_with(api_key="test-key")

    @patch("google.genai.Client")
    @patch.dict(
        "os.environ",
        {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "", "GOOGLE_GENAI_USE_VERTEXAI": ""},
        clear=False,
    )
    def test_no_auth_raises_error(self, mock_client_cls: MagicMock) -> None:
        with pytest.raises(click.ClickException, match="Authentication required"):
            get_gemini_client(api_key=None, use_vertex=False)

    @patch("google.genai.Client")
    def test_vertex_ai_auth(self, mock_client_cls: MagicMock) -> None:
        get_gemini_client(use_vertex=True, project="my-project", location="us-central1")
        mock_client_cls.assert_called_once_with(
            vertexai=True, project="my-project", location="us-central1"
        )

    @patch("google.genai.Client")
    def test_vertex_ai_no_project_raises(self, mock_client_cls: MagicMock) -> None:
        with (
            patch.dict(
                "os.environ",
                {
                    "GOOGLE_GENAI_USE_VERTEXAI": "true",
                    "YT_PROCESS_PROJECT": "",
                    "GOOGLE_CLOUD_PROJECT": "",
                    "GCP_PROJECT": "",
                    "CLOUDSDK_CORE_PROJECT": "",
                },
            ),
            pytest.raises(click.ClickException, match="GCP project"),
        ):
            get_gemini_client(use_vertex=True)

    @patch("google.genai.Client")
    def test_vertex_env_auto_detect(self, mock_client_cls: MagicMock) -> None:
        with patch.dict(
            "os.environ",
            {
                "GOOGLE_GENAI_USE_VERTEXAI": "true",
                "GOOGLE_CLOUD_PROJECT": "env-proj",
            },
        ):
            get_gemini_client()
            call_kwargs = mock_client_cls.call_args[1]
            assert call_kwargs["vertexai"] is True
            assert call_kwargs["project"] == "env-proj"

    @patch("google.genai.Client")
    @patch.dict(
        "os.environ",
        {
            "GEMINI_API_KEY": "",
            "GOOGLE_API_KEY": "env-key",
            "GOOGLE_GENAI_USE_VERTEXAI": "",
        },
    )
    def test_google_api_key_env_fallback(self, mock_client_cls: MagicMock) -> None:
        get_gemini_client()
        mock_client_cls.assert_called_once_with(api_key="env-key")


# ---------------------------------------------------------------------------
# _normalize_timestamp_to_hhmmss
# ---------------------------------------------------------------------------
class TestNormalizeTimestamp:
    def test_mm_ss(self) -> None:
        assert _normalize_timestamp_to_hhmmss("5:30") == "00:05:30"

    def test_hh_mm_ss(self) -> None:
        assert _normalize_timestamp_to_hhmmss("1:05:30") == "01:05:30"

    def test_single_segment(self) -> None:
        assert _normalize_timestamp_to_hhmmss("30") == "30"


# ---------------------------------------------------------------------------
# _format_duration
# ---------------------------------------------------------------------------
class TestFormatDuration:
    def test_zero(self) -> None:
        assert _format_duration(0) == "00:00:00"

    def test_seconds_only(self) -> None:
        assert _format_duration(45) == "00:00:45"

    def test_minutes_and_seconds(self) -> None:
        assert _format_duration(125) == "00:02:05"

    def test_hours(self) -> None:
        assert _format_duration(3661) == "01:01:01"


# ---------------------------------------------------------------------------
# fetch_youtube_chapters
# ---------------------------------------------------------------------------
class TestFetchYoutubeChapters:
    def test_invalid_url_returns_empty(self) -> None:
        assert fetch_youtube_chapters("not-a-youtube-url") == []

    @patch("youtube_gemini_processor.cli.urllib.request.urlopen")
    def test_chapters_parsed(self, mock_urlopen: MagicMock) -> None:
        html = '{"shortDescription":"0:00 Introduction\\n5:30 Main Topic\\n10:00 Conclusion"}'
        mock_resp = MagicMock()
        mock_resp.read.return_value = html.encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        chapters = fetch_youtube_chapters("https://www.youtube.com/watch?v=abc123")
        assert len(chapters) == 3
        assert chapters[0]["title"] == "Introduction"
        assert chapters[0]["start_time"] == "00:00:00"
        assert chapters[1]["start_time"] == "00:05:30"
        assert chapters[2]["end_time"] == ""

    @patch("youtube_gemini_processor.cli.urllib.request.urlopen")
    def test_no_description_returns_empty(self, mock_urlopen: MagicMock) -> None:
        html = "<html>No description field here</html>"
        mock_resp = MagicMock()
        mock_resp.read.return_value = html.encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        assert fetch_youtube_chapters("https://www.youtube.com/watch?v=abc") == []

    @patch("youtube_gemini_processor.cli.urllib.request.urlopen")
    def test_single_chapter_returns_empty(self, mock_urlopen: MagicMock) -> None:
        html = '{"shortDescription":"0:00 Only one chapter"}'
        mock_resp = MagicMock()
        mock_resp.read.return_value = html.encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        assert fetch_youtube_chapters("https://www.youtube.com/watch?v=abc") == []

    @patch(
        "youtube_gemini_processor.cli.urllib.request.urlopen",
        side_effect=Exception("network error"),
    )
    def test_network_error_returns_empty(self, mock_urlopen: MagicMock) -> None:
        assert fetch_youtube_chapters("https://www.youtube.com/watch?v=abc") == []


# ---------------------------------------------------------------------------
# get_video_duration
# ---------------------------------------------------------------------------
class TestGetVideoDuration:
    @patch("youtube_gemini_processor.cli.shutil.which", return_value=None)
    def test_no_ffprobe_returns_none(self, mock_which: MagicMock) -> None:
        assert get_video_duration("/fake/video.mp4") is None

    @patch("youtube_gemini_processor.cli.subprocess.run")
    @patch("youtube_gemini_processor.cli.shutil.which", return_value="/usr/bin/ffprobe")
    def test_success(self, mock_which: MagicMock, mock_run: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "125.5\n"
        mock_run.return_value = mock_result

        assert get_video_duration("/fake/video.mp4") == "00:02:05"

    @patch("youtube_gemini_processor.cli.subprocess.run")
    @patch("youtube_gemini_processor.cli.shutil.which", return_value="/usr/bin/ffprobe")
    def test_nonzero_returncode_returns_none(
        self, mock_which: MagicMock, mock_run: MagicMock
    ) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        assert get_video_duration("/fake/video.mp4") is None

    @patch("youtube_gemini_processor.cli.subprocess.run", side_effect=OSError("fail"))
    @patch("youtube_gemini_processor.cli.shutil.which", return_value="/usr/bin/ffprobe")
    def test_oserror_returns_none(
        self, mock_which: MagicMock, mock_run: MagicMock
    ) -> None:
        assert get_video_duration("/fake/video.mp4") is None


# ---------------------------------------------------------------------------
# get_video_duration_gcs
# ---------------------------------------------------------------------------
class TestGetVideoDurationGcs:
    @patch("youtube_gemini_processor.cli.shutil.which", return_value=None)
    def test_no_ffprobe_returns_none(self, mock_which: MagicMock) -> None:
        assert get_video_duration_gcs("gs://bucket/video.mp4") is None

    @patch("youtube_gemini_processor.cli.shutil.which", return_value="/usr/bin/ffprobe")
    def test_invalid_uri_returns_none(self, mock_which: MagicMock) -> None:
        assert get_video_duration_gcs("not-a-gcs-uri") is None

    @patch("youtube_gemini_processor.cli.subprocess.run")
    @patch("google.auth.transport.requests.Request")
    @patch("google.auth.default")
    @patch("youtube_gemini_processor.cli.shutil.which", return_value="/usr/bin/ffprobe")
    def test_success(
        self,
        mock_which: MagicMock,
        mock_auth_default: MagicMock,
        mock_request: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        mock_creds = MagicMock()
        mock_creds.token = "test-oauth-value"  # noqa: S105
        mock_auth_default.return_value = (mock_creds, "project")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "3661.0\n"
        mock_run.return_value = mock_result

        assert get_video_duration_gcs("gs://bucket/video.mp4") == "01:01:01"

    @patch("google.auth.default", side_effect=Exception("no credentials"))
    @patch("youtube_gemini_processor.cli.shutil.which", return_value="/usr/bin/ffprobe")
    def test_auth_failure_returns_none(
        self, mock_which: MagicMock, mock_auth: MagicMock
    ) -> None:
        assert get_video_duration_gcs("gs://bucket/video.mp4") is None

    @patch("youtube_gemini_processor.cli.subprocess.run")
    @patch("google.auth.transport.requests.Request")
    @patch("google.auth.default")
    @patch("youtube_gemini_processor.cli.shutil.which", return_value="/usr/bin/ffprobe")
    def test_ffprobe_failure_returns_none(
        self,
        mock_which: MagicMock,
        mock_auth_default: MagicMock,
        mock_request: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        mock_creds = MagicMock()
        mock_creds.token = "test-oauth-value"  # noqa: S105
        mock_auth_default.return_value = (mock_creds, "project")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_run.return_value = mock_result

        assert get_video_duration_gcs("gs://bucket/video.mp4") is None

    @patch("youtube_gemini_processor.cli.subprocess.run", side_effect=OSError("fail"))
    @patch("google.auth.transport.requests.Request")
    @patch("google.auth.default")
    @patch("youtube_gemini_processor.cli.shutil.which", return_value="/usr/bin/ffprobe")
    def test_oserror_returns_none(
        self,
        mock_which: MagicMock,
        mock_auth_default: MagicMock,
        mock_request: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        mock_creds = MagicMock()
        mock_creds.token = "test-oauth-value"  # noqa: S105
        mock_auth_default.return_value = (mock_creds, "project")

        assert get_video_duration_gcs("gs://bucket/video.mp4") is None


# ---------------------------------------------------------------------------
# process_video
# ---------------------------------------------------------------------------
class TestProcessVideo:
    def test_success(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "# Analysis\nTitle: My Video\n## Summary\nGreat video."
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 50
        mock_response.usage_metadata = mock_usage
        mock_client.models.generate_content.return_value = mock_response

        analysis = process_video(
            mock_client,
            "https://www.youtube.com/watch?v=abc123",
            "Analyze",
            "gemini-3-flash-preview",
        )

        assert analysis.error is None
        assert analysis.raw_response == mock_response.text
        assert analysis.usage is not None
        assert analysis.usage.input_tokens == 100

    def test_api_error_captured(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API down")

        analysis = process_video(
            mock_client,
            "https://www.youtube.com/watch?v=abc123",
            "Analyze",
        )

        assert analysis.error == "API down"

    def test_with_video_options(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        analysis = process_video(
            mock_client,
            "https://www.youtube.com/watch?v=abc123",
            "Analyze",
            fps=2.0,
            clip_start="0s",
            clip_end="60s",
            media_resolution="MEDIA_RESOLUTION_LOW",
        )

        assert analysis.error is None


# ---------------------------------------------------------------------------
# process_local_file
# ---------------------------------------------------------------------------
class TestProcessLocalFile:
    def test_success(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.name = "files/x"
        mock_uploaded.uri = "https://example.com/files/x"
        mock_uploaded.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_uploaded

        mock_response = MagicMock()
        mock_response.text = "Analysis result"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        analysis = process_local_file(mock_client, str(video), "Analyze")
        assert analysis.error is None
        assert analysis.raw_response == "Analysis result"

    def test_processing_state_waits(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_processing = MagicMock()
        mock_processing.name = "files/x"
        mock_processing.state.name = "PROCESSING"

        mock_active = MagicMock()
        mock_active.name = "files/x"
        mock_active.uri = "https://example.com/files/x"
        mock_active.state.name = "ACTIVE"

        mock_client.files.upload.return_value = mock_processing
        mock_client.files.get.return_value = mock_active

        mock_response = MagicMock()
        mock_response.text = "Done"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        with patch("time.sleep"):
            analysis = process_local_file(mock_client, str(video), "Analyze")

        assert analysis.error is None

    def test_failed_state(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.name = "files/x"
        mock_uploaded.state.name = "FAILED"
        mock_client.files.upload.return_value = mock_uploaded

        analysis = process_local_file(mock_client, str(video), "Analyze")
        assert analysis.error is not None
        assert "failed" in analysis.error.lower() or "File processing" in analysis.error

    def test_verbose_output(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.name = "files/x"
        mock_uploaded.uri = "https://example.com/files/x"
        mock_uploaded.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_uploaded

        mock_response = MagicMock()
        mock_response.text = "Done"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        analysis = process_local_file(mock_client, str(video), "Analyze", verbose=True)
        assert analysis.error is None


# ---------------------------------------------------------------------------
# process_gcs_uri
# ---------------------------------------------------------------------------
class TestProcessGcsUri:
    def test_success(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "GCS analysis"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        analysis = process_gcs_uri(mock_client, "gs://bucket/video.mp4", "Analyze")
        assert analysis.error is None
        assert analysis.raw_response == "GCS analysis"
        assert analysis.url == "gs://bucket/video.mp4"

    def test_verbose(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "GCS analysis"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        analysis = process_gcs_uri(
            mock_client, "gs://bucket/video.mp4", "Analyze", verbose=True
        )
        assert analysis.error is None

    def test_error_captured(self) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("GCS error")

        analysis = process_gcs_uri(mock_client, "gs://bucket/video.mp4", "Analyze")
        assert analysis.error == "GCS error"

    def test_unknown_extension_defaults_mp4(self) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Analysis"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        analysis = process_gcs_uri(mock_client, "gs://bucket/video.xyz", "Analyze")
        assert analysis.error is None


# ---------------------------------------------------------------------------
# _process_single_chapter
# ---------------------------------------------------------------------------
class TestProcessSingleChapter:
    def test_no_start_time_skipped(self, tmp_path: Path) -> None:
        chapter = {
            "segment_number": 1,
            "title": "Intro",
            "start_time": "",
            "end_time": "",
        }
        path, _usage, msg = _process_single_chapter(
            MagicMock(),
            "https://youtube.com/watch?v=x",
            chapter,
            3,
            "Analyze",
            "gemini-3-flash-preview",
            tmp_path,
            format_output_markdown,
            "md",
            False,
            None,
            None,
        )
        assert path is None
        assert "Skipped" in msg

    def test_invalid_timestamps_skipped(self, tmp_path: Path) -> None:
        chapter = {
            "segment_number": 1,
            "title": "Intro",
            "start_time": "bad",
            "end_time": "bad",
        }
        path, _usage, msg = _process_single_chapter(
            MagicMock(),
            "https://youtube.com/watch?v=x",
            chapter,
            3,
            "Analyze",
            "gemini-3-flash-preview",
            tmp_path,
            format_output_markdown,
            "md",
            False,
            None,
            None,
        )
        assert path is None
        assert "invalid" in msg.lower()

    def test_successful_chapter(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Chapter analysis"
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 50
        mock_response.usage_metadata = mock_usage
        mock_client.models.generate_content.return_value = mock_response

        chapter = {
            "segment_number": 1,
            "title": "Intro",
            "start_time": "0:00",
            "end_time": "5:00",
        }
        path, usage, _msg = _process_single_chapter(
            mock_client,
            "https://youtube.com/watch?v=x",
            chapter,
            3,
            "Analyze",
            "gemini-3-flash-preview",
            tmp_path,
            format_output_markdown,
            "md",
            False,
            None,
            None,
        )
        assert path is not None
        assert path.exists()
        assert usage is not None

    def test_api_error_in_chapter(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("fail")

        chapter = {
            "segment_number": 1,
            "title": "Intro",
            "start_time": "0:00",
            "end_time": "5:00",
        }
        path, _usage, msg = _process_single_chapter(
            mock_client,
            "https://youtube.com/watch?v=x",
            chapter,
            3,
            "Analyze",
            "gemini-3-flash-preview",
            tmp_path,
            format_output_markdown,
            "md",
            False,
            None,
            None,
        )
        assert path is None
        assert "Error" in msg

    def test_chapter_with_no_end_time(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        chapter = {
            "segment_number": 3,
            "title": "Last",
            "start_time": "10:00",
            "end_time": "",
        }
        path, _usage, _msg = _process_single_chapter(
            mock_client,
            "https://youtube.com/watch?v=x",
            chapter,
            3,
            "Analyze",
            "gemini-3-flash-preview",
            tmp_path,
            format_output_markdown,
            "md",
            False,
            None,
            None,
        )
        assert path is not None


# ---------------------------------------------------------------------------
# split_youtube_video
# ---------------------------------------------------------------------------
class TestSplitYoutubeVideo:
    def test_processes_chapters(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Chapter analysis"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        chapters = [
            {
                "segment_number": 1,
                "title": "Intro",
                "start_time": "0:00",
                "end_time": "5:00",
            },
            {
                "segment_number": 2,
                "title": "Main",
                "start_time": "5:00",
                "end_time": "10:00",
            },
        ]

        created = split_youtube_video(
            mock_client,
            "https://youtube.com/watch?v=abc123",
            chapters,
            "Analyze",
            "gemini-3-flash-preview",
            tmp_path,
            "markdown",
            max_workers=1,
        )
        assert len(created) == 2


# ---------------------------------------------------------------------------
# split_video (verbose branch)
# ---------------------------------------------------------------------------
class TestSplitVideoVerbose:
    @patch("youtube_gemini_processor.cli.subprocess.run")
    @patch("youtube_gemini_processor.cli.shutil.which", return_value="/usr/bin/ffmpeg")
    def test_verbose_output(
        self, mock_which: MagicMock, mock_run: MagicMock, tmp_path: Path
    ) -> None:
        source = tmp_path / "video.mp4"
        source.write_bytes(b"fake")

        def create_output(cmd, **_kwargs):
            out_path = Path(cmd[-1])
            out_path.write_bytes(b"segment")
            result = MagicMock()
            result.returncode = 0
            return result

        mock_run.side_effect = create_output

        segments = [
            {
                "segment_number": 1,
                "start_time": "00:00:00",
                "end_time": "00:10:00",
                "title": "Intro",
            },
        ]
        created = split_video(str(source), segments, tmp_path, verbose=True)
        assert len(created) == 1


# ---------------------------------------------------------------------------
# _handle_output
# ---------------------------------------------------------------------------
class TestHandleOutput:
    def test_output_to_dir(self, tmp_path: Path) -> None:
        _handle_output(
            "formatted content",
            "https://youtube.com/watch?v=abc123",
            "md",
            tmp_path,
            None,
            True,
            False,
        )
        files = list(tmp_path.glob("*.md"))
        assert len(files) == 1
        assert files[0].read_text() == "formatted content"

    def test_output_to_file(self, tmp_path: Path) -> None:
        out_file = tmp_path / "output.md"
        _handle_output("content", "input", "md", None, out_file, False, False)
        assert out_file.read_text() == "content"

    def test_output_to_stdout(self) -> None:
        # No file or dir - should print to stdout (no error)
        _handle_output("content", "input", "md", None, None, False, False)

    def test_verbose_dir(self, tmp_path: Path) -> None:
        _handle_output(
            "content", "https://youtube.com/watch?v=x", "md", tmp_path, None, True, True
        )

    def test_verbose_file(self, tmp_path: Path) -> None:
        out_file = tmp_path / "out.md"
        _handle_output("content", "input", "md", None, out_file, False, True)
        assert out_file.read_text() == "content"


# ---------------------------------------------------------------------------
# format_segments_markdown / json with usage
# ---------------------------------------------------------------------------
class TestFormatSegmentsWithUsage:
    def test_markdown_with_usage(self) -> None:
        from youtube_gemini_processor.cli import UsageStats

        analysis = VideoAnalysis(
            url="/video.mp4",
            processed_at="2024-01-01",
            model="gemini-3-flash-preview",
            usage=UsageStats(100, 50, 150, 0.001, 0.002, 0.003),
        )
        segments = [
            {
                "segment_number": 1,
                "start_time": "00:00:00",
                "end_time": "00:05:00",
                "title": "Intro",
                "speaker": "",
                "summary": "Opening",
            },
        ]
        result = format_segments_markdown(analysis, segments)
        assert "**Usage**" in result
        assert "**Cost**" in result

    def test_json_with_usage(self) -> None:
        from youtube_gemini_processor.cli import UsageStats

        analysis = VideoAnalysis(
            url="/video.mp4",
            processed_at="2024-01-01",
            model="gemini-3-flash-preview",
            usage=UsageStats(100, 50, 150, 0.001, 0.002, 0.003),
        )
        segments = [
            {
                "segment_number": 1,
                "start_time": "00:00:00",
                "end_time": "00:05:00",
                "title": "Intro",
                "speaker": "A",
                "summary": "Hi",
            },
        ]
        result = format_segments_json(analysis, segments)
        data = json.loads(result)
        assert data["usage"]["input_tokens"] == 100


# ---------------------------------------------------------------------------
# CLI integration: main() branches
# ---------------------------------------------------------------------------
class TestMainBranches:
    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_upload_only_no_input_errors(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value = MagicMock()
        runner = CliRunner()
        result = runner.invoke(main, ["--upload-only", "--api-key", "k"])
        assert result.exit_code != 0

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_upload_only_with_batch_errors(
        self, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        batch = tmp_path / "urls.txt"
        batch.write_text("https://youtube.com/watch?v=x")
        mock_get_client.return_value = MagicMock()
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--batch", str(batch), "--upload-only", "--api-key", "k"],
        )
        assert result.exit_code != 0

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_upload_only_processing_wait(
        self, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_processing = MagicMock()
        mock_processing.name = "files/x"
        mock_processing.uri = "https://example.com/files/x"
        mock_processing.state.name = "PROCESSING"

        mock_active = MagicMock()
        mock_active.name = "files/x"
        mock_active.uri = "https://example.com/files/x"
        mock_active.state.name = "ACTIVE"
        mock_active.expiration_time = "2026-04-01"

        mock_client.files.upload.return_value = mock_processing
        mock_client.files.get.return_value = mock_active
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        with patch("youtube_gemini_processor.cli.time.sleep"):
            result = runner.invoke(
                main,
                [str(video), "--upload-only", "-v", "--api-key", "k"],
            )
        assert result.exit_code == 0
        assert "files/x" in result.output

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_upload_only_failed_state(
        self, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_file = MagicMock()
        mock_file.name = "files/x"
        mock_file.state.name = "FAILED"
        mock_client.files.upload.return_value = mock_file
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(main, [str(video), "--upload-only", "--api-key", "k"])
        assert result.exit_code != 0

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    @patch("youtube_gemini_processor.cli.fetch_youtube_chapters")
    def test_split_youtube_no_chapters_errors(
        self, mock_chapters: MagicMock, mock_get_client: MagicMock
    ) -> None:
        mock_chapters.return_value = []
        mock_get_client.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "https://youtube.com/watch?v=abc",
                "--split",
                "--prompt",
                "custom",
                "--api-key",
                "k",
            ],
        )
        assert result.exit_code != 0
        assert "chapter" in result.output.lower()

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    @patch("youtube_gemini_processor.cli.fetch_youtube_chapters")
    def test_split_youtube_with_chapters(
        self, mock_chapters: MagicMock, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        mock_chapters.return_value = [
            {
                "segment_number": 1,
                "title": "Intro",
                "start_time": "00:00:00",
                "end_time": "00:05:00",
            },
            {
                "segment_number": 2,
                "title": "Main",
                "start_time": "00:05:00",
                "end_time": "",
            },
        ]
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Analysis"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "https://youtube.com/watch?v=abc",
                "--split",
                "--prompt",
                "custom",
                "--api-key",
                "k",
                "-o",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_gcs_uri_input(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "GCS analysis"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["gs://bucket/video.mp4", "--api-key", "k"],
        )
        assert result.exit_code == 0
        assert "Video Analysis" in result.output

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_segments_mode_json_format(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            [
                {
                    "segment_number": 1,
                    "start_time": "00:00:00",
                    "end_time": "00:05:00",
                    "title": "Intro",
                    "speaker": "A",
                    "summary": "Hi",
                },
            ]
        )
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "https://youtube.com/watch?v=abc",
                "--mode",
                "segments",
                "--format",
                "json",
                "--api-key",
                "k",
            ],
        )
        assert result.exit_code == 0
        # Output has progress bar prefix before JSON, extract JSON part
        output = result.output
        json_start = output.find("{")
        assert json_start != -1
        data = json.loads(output[json_start:])
        assert "segments" in data

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    @patch("youtube_gemini_processor.cli.get_video_duration")
    def test_segments_mode_local_file_with_duration(
        self, mock_duration: MagicMock, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        mock_duration.return_value = "01:30:00"
        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.name = "files/x"
        mock_uploaded.uri = "https://example.com/files/x"
        mock_uploaded.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_uploaded

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            [
                {
                    "segment_number": 1,
                    "start_time": "00:00:00",
                    "end_time": "01:30:00",
                    "title": "Full",
                    "speaker": "A",
                    "summary": "All",
                },
            ]
        )
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [str(video), "--mode", "segments", "-v", "--api-key", "k"],
        )
        assert result.exit_code == 0

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    @patch("youtube_gemini_processor.cli.split_video")
    @patch("youtube_gemini_processor.cli.get_video_duration", return_value=None)
    def test_segments_split_local_file(
        self,
        mock_duration: MagicMock,
        mock_split: MagicMock,
        mock_get_client: MagicMock,
        tmp_path: Path,
    ) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        mock_split.return_value = [tmp_path / "seg1.mp4"]
        mock_client = MagicMock()
        mock_uploaded = MagicMock()
        mock_uploaded.name = "files/x"
        mock_uploaded.uri = "https://example.com/files/x"
        mock_uploaded.state.name = "ACTIVE"
        mock_client.files.upload.return_value = mock_uploaded

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            [
                {
                    "segment_number": 1,
                    "start_time": "00:00:00",
                    "end_time": "00:10:00",
                    "title": "Intro",
                    "speaker": "A",
                    "summary": "Hi",
                },
            ]
        )
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [str(video), "--mode", "segments", "--split", "--api-key", "k"],
        )
        assert result.exit_code == 0
        mock_split.assert_called_once()

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_segments_split_non_local_warns(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            [
                {
                    "segment_number": 1,
                    "start_time": "00:00:00",
                    "end_time": "00:05:00",
                    "title": "Intro",
                    "speaker": "A",
                    "summary": "Hi",
                },
            ]
        )
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "https://youtube.com/watch?v=abc",
                "--mode",
                "segments",
                "--split",
                "--api-key",
                "k",
            ],
        )
        assert result.exit_code == 0

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_batch_with_failures_verbose(
        self, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        batch = tmp_path / "urls.txt"
        batch.write_text(
            "https://youtube.com/watch?v=v1\nhttps://youtube.com/watch?v=v2\n"
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        mock_client = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("API error")
            mock_resp = MagicMock()
            mock_resp.text = "OK"
            mock_resp.usage_metadata = None
            return mock_resp

        mock_client.models.generate_content.side_effect = side_effect
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--batch",
                str(batch),
                "-o",
                str(output_dir),
                "-v",
                "--api-key",
                "k",
            ],
        )
        assert "1 successful" in result.output or "1 failed" in result.output

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    @patch("youtube_gemini_processor.cli.get_video_duration_gcs")
    def test_segments_gcs_duration_detection(
        self, mock_duration: MagicMock, mock_get_client: MagicMock
    ) -> None:
        mock_duration.return_value = "00:30:00"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            [
                {
                    "segment_number": 1,
                    "start_time": "00:00:00",
                    "end_time": "00:30:00",
                    "title": "All",
                    "speaker": "A",
                    "summary": "Everything",
                },
            ]
        )
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["gs://bucket/video.mp4", "--mode", "segments", "--api-key", "k"],
        )
        assert result.exit_code == 0
        mock_duration.assert_called_once()

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_split_local_without_segments_errors(
        self, mock_get_client: MagicMock, tmp_path: Path
    ) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        mock_get_client.return_value = MagicMock()

        runner = CliRunner()
        result = runner.invoke(
            main,
            [str(video), "--split", "--api-key", "k"],
        )
        assert result.exit_code != 0
        assert "segments" in result.output.lower()

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_custom_prompt_overrides_mode(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Custom analysis"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "https://youtube.com/watch?v=abc",
                "--prompt",
                "Focus on cats",
                "--api-key",
                "k",
            ],
        )
        assert result.exit_code == 0

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_split_youtube_no_output_dir(
        self, mock_get_client: MagicMock, tmp_path: Path, monkeypatch
    ) -> None:
        """Test --split YouTube without -o uses auto-generated dir name."""
        monkeypatch.chdir(tmp_path)

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Analysis"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        with patch("youtube_gemini_processor.cli.fetch_youtube_chapters") as mock_ch:
            mock_ch.return_value = [
                {
                    "segment_number": 1,
                    "title": "Intro",
                    "start_time": "00:00:00",
                    "end_time": "00:05:00",
                },
            ]
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "https://youtube.com/watch?v=abc123",
                    "--split",
                    "--prompt",
                    "custom",
                    "--api-key",
                    "k",
                ],
            )
        assert result.exit_code == 0

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_output_to_file(self, mock_get_client: MagicMock, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Analysis"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        out_file = tmp_path / "output.md"
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "https://youtube.com/watch?v=abc",
                "-o",
                str(out_file),
                "--api-key",
                "k",
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()


# ---------------------------------------------------------------------------
# parse_clip_range edge cases (>2 hyphens and else branch)
# ---------------------------------------------------------------------------
class TestParseClipRangeEdge:
    def test_hh_mm_ss_range(self) -> None:
        """Test HH:MM:SS-HH:MM:SS which splits into >2 parts on '-'."""
        from youtube_gemini_processor.cli import parse_clip_range

        start, end = parse_clip_range("0:01:30-0:05:00")
        assert start == "90s"
        assert end == "300s"


# ---------------------------------------------------------------------------
# get_safe_filename for GCS URIs
# ---------------------------------------------------------------------------
class TestGetSafeFilenameGcs:
    def test_gcs_uri(self) -> None:
        from youtube_gemini_processor.cli import get_safe_filename

        assert get_safe_filename("gs://bucket/my_video.mp4") == "my_video"


# ---------------------------------------------------------------------------
# process_files_api_ref generic exception branch
# ---------------------------------------------------------------------------
class TestProcessFilesApiRefGenericError:
    def test_generic_exception_captured(self) -> None:
        mock_client = MagicMock()
        mock_file = MagicMock()
        mock_file.name = "files/abc123"
        mock_file.uri = "https://generativelanguage.googleapis.com/v1beta/files/abc123"
        mock_file.state.name = "ACTIVE"
        mock_file.mime_type = "video/mp4"
        mock_file.display_name = "test.mp4"
        mock_client.files.get.return_value = mock_file
        # Make generate_content raise a non-ClickException error
        mock_client.models.generate_content.side_effect = RuntimeError("unexpected")

        from youtube_gemini_processor.cli import process_files_api_ref

        analysis = process_files_api_ref(mock_client, "files/abc123", "Analyze")
        assert analysis.error == "unexpected"


# ---------------------------------------------------------------------------
# process_files_api_ref verbose branches
# ---------------------------------------------------------------------------
class TestProcessFilesApiRefVerbose:
    def test_verbose_output(self) -> None:
        from youtube_gemini_processor.cli import process_files_api_ref

        mock_client = MagicMock()
        mock_file = MagicMock()
        mock_file.name = "files/abc123"
        mock_file.uri = "https://generativelanguage.googleapis.com/v1beta/files/abc123"
        mock_file.state.name = "ACTIVE"
        mock_file.mime_type = "video/webm"
        mock_file.display_name = "test.webm"
        mock_client.files.get.return_value = mock_file

        mock_response = MagicMock()
        mock_response.text = "Analysis"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        analysis = process_files_api_ref(
            mock_client, "files/abc123", "Analyze", verbose=True
        )
        assert analysis.error is None


# ---------------------------------------------------------------------------
# process_local_file verbose + processing wait
# ---------------------------------------------------------------------------
class TestProcessLocalFileProcessingWait:
    def test_processing_then_active_verbose(self, tmp_path: Path) -> None:
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        mock_client = MagicMock()
        mock_processing = MagicMock()
        mock_processing.name = "files/x"
        mock_processing.state.name = "PROCESSING"

        mock_active = MagicMock()
        mock_active.name = "files/x"
        mock_active.uri = "https://example.com/files/x"
        mock_active.state.name = "ACTIVE"

        mock_client.files.upload.return_value = mock_processing
        mock_client.files.get.return_value = mock_active

        mock_response = MagicMock()
        mock_response.text = "Done"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.return_value = mock_response

        with patch("time.sleep"):
            analysis = process_local_file(
                mock_client, str(video), "Analyze", verbose=True
            )
        assert analysis.error is None


# ---------------------------------------------------------------------------
# split_youtube_video with usage stats
# ---------------------------------------------------------------------------
class TestSplitYoutubeVideoWithUsage:
    def test_accumulates_usage(self, tmp_path: Path) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Analysis"
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 100
        mock_usage.candidates_token_count = 50
        mock_response.usage_metadata = mock_usage
        mock_client.models.generate_content.return_value = mock_response

        chapters = [
            {
                "segment_number": 1,
                "title": "Intro",
                "start_time": "0:00",
                "end_time": "5:00",
            },
        ]

        created = split_youtube_video(
            mock_client,
            "https://youtube.com/watch?v=abc",
            chapters,
            "Analyze",
            "gemini-3-flash-preview",
            tmp_path,
            "markdown",
            max_workers=1,
        )
        assert len(created) == 1


# ---------------------------------------------------------------------------
# _process_single_chapter: analysis.error branch
# ---------------------------------------------------------------------------
class TestProcessSingleChapterAnalysisError:
    def test_analysis_error_returned(self, tmp_path: Path) -> None:
        """When process_video returns analysis with error, chapter reports error."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Content"
        mock_response.usage_metadata = None
        mock_client.models.generate_content.side_effect = Exception("API fail")

        chapter = {
            "segment_number": 1,
            "title": "Intro",
            "start_time": "0:00",
            "end_time": "5:00",
        }
        path, _usage, msg = _process_single_chapter(
            mock_client,
            "https://youtube.com/watch?v=x",
            chapter,
            3,
            "Analyze",
            "gemini-3-flash-preview",
            tmp_path,
            format_output_markdown,
            "md",
            False,
            None,
            None,
        )
        assert path is None
        assert "Error" in msg
