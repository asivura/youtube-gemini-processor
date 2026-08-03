"""Tests for model refresh, cost accuracy, Vertex transport, and robustness."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from youtube_gemini_processor import cli as cli_module
from youtube_gemini_processor.cli import (
    AUDIO_PROMPTS,
    DEFAULT_MODEL,
    INLINE_MAX_BYTES,
    MODEL_PRICING,
    PROMPTS,
    SUGGESTED_MODELS,
    UsageStats,
    VideoAnalysis,
    _as_int,
    _build_vertex_local_part,
    _decode_json_string,
    _require_developer_api,
    build_duration_line,
    build_inline_media_part,
    calculate_cost,
    extract_title,
    extract_usage,
    extract_video_id,
    fetch_youtube_duration,
    format_output_json,
    format_output_markdown,
    gcs_object_name,
    generate_with_retry,
    is_vertex_client,
    main,
    media_kind_for_input,
    select_prompt,
    upload_to_gcs,
    validate_youtube_url,
)


def _usage_meta(prompt=0, candidates=0, thoughts=0, cached=0, audio=None) -> MagicMock:
    """Build a usage_metadata stub with real ints, not mock attributes."""
    meta = MagicMock()
    meta.prompt_token_count = prompt
    meta.candidates_token_count = candidates
    meta.thoughts_token_count = thoughts
    meta.cached_content_token_count = cached
    if audio is None:
        meta.prompt_tokens_details = []
    else:
        detail = MagicMock()
        detail.modality.name = "AUDIO"
        detail.token_count = audio
        meta.prompt_tokens_details = [detail]
    return meta


class TestModelRegistry:
    """The model list should track what Google actually ships."""

    def test_default_is_the_frontier_pro_model(self) -> None:
        assert DEFAULT_MODEL == "gemini-3.1-pro-preview"
        assert DEFAULT_MODEL in MODEL_PRICING

    @pytest.mark.parametrize(
        "model",
        [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
    )
    def test_newer_models_are_priced(self, model: str) -> None:
        assert model in MODEL_PRICING
        assert MODEL_PRICING[model]["input"]
        assert MODEL_PRICING[model]["output"]

    def test_suggested_models_all_have_pricing(self) -> None:
        for model in SUGGESTED_MODELS:
            assert model in MODEL_PRICING

    def test_arbitrary_model_id_is_accepted(self) -> None:
        """--model must not be a closed list, or new models need a code edit."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        # A Choice would render the options inline; a free string must not.
        assert "[gemini-3.1-pro-preview|" not in result.output


class TestCostAccuracy:
    """Gemini bills thinking tokens at the output rate."""

    def test_thinking_tokens_are_billed_as_output(self) -> None:
        without = calculate_cost("gemini-3.1-pro-preview", 1000, 100)
        with_thinking = calculate_cost(
            "gemini-3.1-pro-preview", 1000, 100, thoughts_tokens=900
        )
        assert with_thinking.output_tokens == 1000
        assert with_thinking.thoughts_tokens == 900
        assert with_thinking.output_cost == pytest.approx(without.output_cost * 10)

    def test_total_tokens_includes_thinking(self) -> None:
        stats = calculate_cost("gemini-3.6-flash", 500, 100, thoughts_tokens=400)
        assert stats.total_tokens == 1000

    def test_audio_tokens_bill_at_the_audio_rate(self) -> None:
        """gemini-2.5-flash-lite charges $0.30/1M audio vs $0.10/1M otherwise."""
        stats = calculate_cost(
            "gemini-2.5-flash-lite", 1_000_000, 0, audio_input_tokens=1_000_000
        )
        assert stats.input_cost == pytest.approx(0.30)

    def test_mixed_modality_input_splits_rates(self) -> None:
        stats = calculate_cost(
            "gemini-2.5-flash-lite", 1_000_000, 0, audio_input_tokens=400_000
        )
        # 400k audio @ $0.30/M + 600k text @ $0.10/M
        assert stats.input_cost == pytest.approx(0.12 + 0.06)

    def test_model_without_audio_rate_uses_the_flat_rate(self) -> None:
        stats = calculate_cost(
            "gemini-3.6-flash", 1_000_000, 0, audio_input_tokens=1_000_000
        )
        assert stats.input_cost == pytest.approx(1.50)

    def test_long_context_reprices_all_tokens(self) -> None:
        """Long-context pricing is a cliff, not a graduated bracket.

        Google bills "prompts > 200k tokens" at the high rate for every token,
        and picks the output rate from the prompt size. Billing only the excess
        at the high rate under-reports the bill by ~19% on this input.
        """
        stats = calculate_cost("gemini-2.5-pro", 1_000_000, 100_000)
        assert stats.input_cost == pytest.approx(2.50)  # all 1M @ $2.50/M
        assert stats.output_cost == pytest.approx(1.50)  # all 100k @ $15/M
        assert stats.total_cost == pytest.approx(4.00)

    def test_short_context_uses_base_rates(self) -> None:
        stats = calculate_cost("gemini-2.5-pro", 100_000, 10_000)
        assert stats.input_cost == pytest.approx(0.125)  # $1.25/M
        assert stats.output_cost == pytest.approx(0.10)  # $10/M

    def test_thinking_tokens_do_not_trigger_long_context(self) -> None:
        """The threshold keys off the prompt, not total or output tokens."""
        stats = calculate_cost("gemini-2.5-pro", 1_000, 10_000, thoughts_tokens=500_000)
        assert stats.output_cost == pytest.approx(510_000 / 1_000_000 * 10.00)

    def test_cached_tokens_are_reported(self) -> None:
        stats = calculate_cost("gemini-3.6-flash", 1000, 10, cached_tokens=800)
        assert stats.cached_tokens == 800


class TestExtractUsage:
    """Usage extraction has to survive partial and odd API responses."""

    def test_reads_all_fields(self) -> None:
        response = MagicMock()
        response.usage_metadata = _usage_meta(
            prompt=1000, candidates=50, thoughts=200, cached=100, audio=300
        )
        usage = extract_usage(response, "gemini-3.6-flash")
        assert usage is not None
        assert usage.input_tokens == 1000
        assert usage.output_tokens == 250
        assert usage.thoughts_tokens == 200
        assert usage.audio_input_tokens == 300
        assert usage.cached_tokens == 100

    def test_missing_usage_metadata_returns_none(self) -> None:
        response = MagicMock()
        response.usage_metadata = None
        assert extract_usage(response, "gemini-3.6-flash") is None

    def test_non_integer_fields_do_not_poison_arithmetic(self) -> None:
        """A bare MagicMock field must read as 0, not blow up formatting."""
        response = MagicMock()
        response.usage_metadata = MagicMock()
        usage = extract_usage(response, "gemini-3.6-flash")
        assert usage is not None
        assert usage.total_tokens == 0

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(5, 5), (5.9, 5), (None, 0), ("7", 0), (True, 0), (MagicMock(), 0)],
    )
    def test_as_int_coercion(self, value, expected: int) -> None:
        assert _as_int(value) == expected


class TestUnknownModelPricing:
    def test_costs_are_absent_not_wrong(self) -> None:
        stats = calculate_cost("some-future-model", 1000, 100)
        assert stats.pricing_known is False
        assert stats.total_cost == 0.0

    def test_markdown_says_unknown(self) -> None:
        analysis = VideoAnalysis(url="u", raw_response="body")
        analysis.usage = calculate_cost("some-future-model", 1000, 100)
        assert "unknown" in format_output_markdown(analysis).lower()

    def test_json_nulls_costs(self) -> None:
        analysis = VideoAnalysis(url="u", raw_response="body")
        analysis.usage = calculate_cost("some-future-model", 1000, 100)
        payload = json.loads(format_output_json(analysis))
        assert payload["usage"]["pricing_known"] is False
        assert payload["usage"]["total_cost_usd"] is None
        assert payload["usage"]["total_tokens"] == 1100


class TestVertexDetection:
    def test_real_vertex_flag(self) -> None:
        client = MagicMock()
        client.vertexai = True
        assert is_vertex_client(client) is True

    def test_developer_api_client(self) -> None:
        client = MagicMock()
        client.vertexai = False
        assert is_vertex_client(client) is False

    def test_truthy_non_bool_is_not_vertex(self) -> None:
        """A stub attribute must not silently route down the Vertex path."""
        assert is_vertex_client(MagicMock()) is False

    def test_require_developer_api_raises_on_vertex(self) -> None:
        client = MagicMock()
        client.vertexai = True
        with pytest.raises(click.ClickException) as exc:
            _require_developer_api(client, "--upload-only")
        assert "--gcs-bucket" in str(exc.value)

    def test_require_developer_api_passes_otherwise(self) -> None:
        client = MagicMock()
        client.vertexai = False
        _require_developer_api(client, "--upload-only")


class TestVertexLocalTransport:
    """Vertex has no Files API; local files go inline or via GCS."""

    def test_small_file_goes_inline(self, tmp_path: Path) -> None:
        media = tmp_path / "clip.m4a"
        media.write_bytes(b"\x00" * 1024)
        part = _build_vertex_local_part(
            media,
            "audio/mp4",
            "audio",
            file_size=1024,
            gcs_bucket=None,
            verbose=False,
            fps=None,
            clip_start=None,
            clip_end=None,
        )
        assert part.inline_data is not None
        assert part.inline_data.mime_type == "audio/mp4"
        assert part.file_data is None

    def test_oversized_file_without_bucket_explains_options(
        self, tmp_path: Path
    ) -> None:
        media = tmp_path / "big.mp4"
        media.write_bytes(b"")
        with pytest.raises(click.ClickException) as exc:
            _build_vertex_local_part(
                media,
                "video/mp4",
                "video",
                file_size=INLINE_MAX_BYTES + 1,
                gcs_bucket=None,
                verbose=False,
                fps=None,
                clip_start=None,
                clip_end=None,
            )
        message = str(exc.value)
        assert "--gcs-bucket" in message
        assert "--clip" in message
        assert "GEMINI_API_KEY" in message

    def test_small_file_stays_inline_even_with_a_bucket_configured(
        self, tmp_path: Path
    ) -> None:
        """Staging persists a permanent copy; a small file must not pay that."""
        media = tmp_path / "memo.m4a"
        media.write_bytes(b"\x00" * 2048)
        with patch("youtube_gemini_processor.cli.upload_to_gcs") as mock_upload:
            part = _build_vertex_local_part(
                media,
                "audio/mp4",
                "audio",
                file_size=2048,
                gcs_bucket="my-bucket",
                verbose=False,
                fps=None,
                clip_start=None,
                clip_end=None,
            )
        mock_upload.assert_not_called()
        assert part.inline_data is not None

    def test_bucket_stages_to_gcs(self, tmp_path: Path) -> None:
        media = tmp_path / "big.mp4"
        media.write_bytes(b"")
        with patch(
            "youtube_gemini_processor.cli.upload_to_gcs",
            return_value="gs://bucket/big.mp4",
        ) as mock_upload:
            part = _build_vertex_local_part(
                media,
                "video/mp4",
                "video",
                file_size=INLINE_MAX_BYTES + 1,
                gcs_bucket="bucket",
                verbose=False,
                fps=None,
                clip_start=None,
                clip_end=None,
            )
        mock_upload.assert_called_once()
        assert part.file_data.file_uri == "gs://bucket/big.mp4"

    def test_inline_part_carries_clip_metadata(self) -> None:
        part = build_inline_media_part(
            b"data", "video/mp4", kind="video", fps=2.0, clip_start="10s"
        )
        assert part.video_metadata.fps == 2.0
        assert part.video_metadata.start_offset == "10s"

    def test_audio_inline_part_has_no_fps(self) -> None:
        part = build_inline_media_part(b"data", "audio/mp4", kind="audio", fps=2.0)
        assert part.video_metadata is None


class TestUploadToGcs:
    def test_missing_gcloud_raises(self, tmp_path: Path) -> None:
        with (
            patch("youtube_gemini_processor.cli.shutil.which", return_value=None),
            pytest.raises(click.ClickException, match="gcloud CLI"),
        ):
            upload_to_gcs(tmp_path / "f.mp4", "bucket")

    def test_strips_gs_prefix_from_bucket(self, tmp_path: Path) -> None:
        media = tmp_path / "f.mp4"
        media.write_bytes(b"")
        result = MagicMock(returncode=0, stderr="")
        with (
            patch("youtube_gemini_processor.cli.shutil.which", return_value="/g"),
            patch(
                "youtube_gemini_processor.cli.subprocess.run", return_value=result
            ) as run,
        ):
            uri = upload_to_gcs(media, "gs://my-bucket/")
        assert uri.startswith("gs://my-bucket/yt-process/")
        assert uri.endswith("/f.mp4")
        assert run.call_args[0][0][-1] == uri

    def test_failed_upload_raises(self, tmp_path: Path) -> None:
        media = tmp_path / "f.mp4"
        media.write_bytes(b"")
        result = MagicMock(returncode=1, stderr="permission denied")
        with (
            patch("youtube_gemini_processor.cli.shutil.which", return_value="/g"),
            patch("youtube_gemini_processor.cli.subprocess.run", return_value=result),
            pytest.raises(click.ClickException, match="permission denied"),
        ):
            upload_to_gcs(media, "bucket")


class TestGcsObjectNaming:
    """A bare basename lets one input be analyzed against another's bytes."""

    def test_same_basename_different_dirs_do_not_collide(self, tmp_path: Path) -> None:
        a = tmp_path / "monday" / "recording.m4a"
        b = tmp_path / "tuesday" / "recording.m4a"
        assert gcs_object_name(a) != gcs_object_name(b)

    def test_name_is_deterministic(self, tmp_path: Path) -> None:
        """Re-running the same file reuses one object instead of littering."""
        media = tmp_path / "a.m4a"
        assert gcs_object_name(media) == gcs_object_name(media)

    def test_basename_is_preserved_for_readability(self, tmp_path: Path) -> None:
        assert gcs_object_name(tmp_path / "standup.m4a").endswith("/standup.m4a")

    def test_empty_bucket_is_rejected(self, tmp_path: Path) -> None:
        media = tmp_path / "f.mp4"
        media.write_bytes(b"")
        with (
            patch("youtube_gemini_processor.cli.shutil.which", return_value="/g"),
            pytest.raises(click.ClickException, match="not a usable bucket"),
        ):
            upload_to_gcs(media, "gs://")


class TestBatchFailureIsolation:
    """One bad input must not discard an already-billed batch."""

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_one_raising_input_does_not_lose_the_others(
        self, mock_client: MagicMock, tmp_path: Path
    ) -> None:
        client = MagicMock()
        client.vertexai = False
        response = MagicMock()
        response.text = "body"
        response.usage_metadata = _usage_meta(prompt=10, candidates=5)
        response.candidates = []
        client.models.generate_content.return_value = response
        mock_client.return_value = client

        batch = tmp_path / "inputs.txt"
        batch.write_text(
            "https://youtube.com/watch?v=aaa\n"
            "https://youtube.com/watch?v=bbb\n"
            "https://youtube.com/watch?v=ccc\n"
        )
        outdir = tmp_path / "out"

        real = cli_module.build_duration_line

        def explode(video_input, **kwargs):
            if "bbb" in video_input:
                raise RuntimeError("ffprobe blew up")
            return real(video_input, **kwargs)

        with patch("youtube_gemini_processor.cli.build_duration_line", explode):
            result = CliRunner().invoke(
                main,
                ["--batch", str(batch), "-o", str(outdir), "--api-key", "k"],
            )

        # The failure is reported and the exit code reflects it...
        assert result.exit_code == 1
        # ...but the two healthy inputs still produced their documents.
        written = sorted(p.name for p in outdir.glob("*.md"))
        assert len(written) == 3, written


class TestRetry:
    def test_returns_first_success(self) -> None:
        client = MagicMock()
        client.models.generate_content.return_value = "ok"
        assert generate_with_retry(client, model="m", contents=[], config=None) == "ok"
        assert client.models.generate_content.call_count == 1

    def test_retries_rate_limit_then_succeeds(self) -> None:
        error = Exception("429 RESOURCE_EXHAUSTED")
        error.code = 429
        client = MagicMock()
        client.models.generate_content.side_effect = [error, error, "ok"]
        slept: list[float] = []
        result = generate_with_retry(
            client, model="m", contents=[], config=None, sleep=slept.append
        )
        assert result == "ok"
        # Exponential, but jittered so parallel workers do not retry in lockstep.
        assert len(slept) == 2
        assert 1.5 <= slept[0] <= 2.5
        assert 3.0 <= slept[1] <= 5.0
        assert slept[1] > slept[0]

    def test_explicit_status_code_overrides_message_text(self) -> None:
        """A 400 whose body mentions UNAVAILABLE must not be retried.

        Retrying re-sends the entire inline payload, so a false positive here
        is expensive as well as pointless.
        """
        error = Exception("400 INVALID_ARGUMENT: service UNAVAILABLE in region")
        error.code = 400
        client = MagicMock()
        client.models.generate_content.side_effect = error
        with pytest.raises(Exception, match="INVALID_ARGUMENT"):
            generate_with_retry(
                client, model="m", contents=[], config=None, sleep=lambda _: None
            )
        assert client.models.generate_content.call_count == 1

    def test_gives_up_after_budget(self) -> None:
        error = Exception("503 UNAVAILABLE")
        client = MagicMock()
        client.models.generate_content.side_effect = error
        with pytest.raises(Exception, match="UNAVAILABLE"):
            generate_with_retry(
                client,
                model="m",
                contents=[],
                config=None,
                max_retries=2,
                sleep=lambda _: None,
            )
        assert client.models.generate_content.call_count == 3

    def test_does_not_retry_client_errors(self) -> None:
        error = Exception("400 INVALID_ARGUMENT")
        error.code = 400
        client = MagicMock()
        client.models.generate_content.side_effect = error
        with pytest.raises(Exception, match="INVALID_ARGUMENT"):
            generate_with_retry(
                client, model="m", contents=[], config=None, sleep=lambda _: None
            )
        assert client.models.generate_content.call_count == 1


class TestAudioHandling:
    @pytest.mark.parametrize("mode", ["comprehensive", "concise", "transcript"])
    def test_audio_prompts_request_no_visual_sections(self, mode: str) -> None:
        """The video prompts' visual output sections invite invented slides."""
        prompt = AUDIO_PROMPTS[mode]
        assert "**Visual Content**" not in prompt
        assert "## All Visual Content" not in prompt
        assert "Text on screen" not in prompt

    @pytest.mark.parametrize("mode", ["comprehensive", "concise", "transcript"])
    def test_audio_prompts_state_the_medium(self, mode: str) -> None:
        prompt = AUDIO_PROMPTS[mode].lower()
        assert "audio" in prompt
        assert any(
            phrase in prompt
            for phrase in ("audio only", "no video", "never describe visual")
        )

    @pytest.mark.parametrize("mode", ["comprehensive", "concise", "transcript"])
    def test_video_prompts_still_request_visuals(self, mode: str) -> None:
        """Guard against the audio rewrite leaking into the video path."""
        assert "isual" in PROMPTS[mode]

    @pytest.mark.parametrize("mode", list(PROMPTS))
    def test_every_mode_has_an_audio_variant(self, mode: str) -> None:
        assert mode in AUDIO_PROMPTS
        assert "{duration_line}" in AUDIO_PROMPTS[mode]

    def test_select_prompt_routes_by_kind(self) -> None:
        assert select_prompt("transcript", "audio") == AUDIO_PROMPTS["transcript"]
        assert select_prompt("transcript", "video") == PROMPTS["transcript"]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("meeting.m4a", "audio"),
            ("meeting.mp3", "audio"),
            ("gs://b/call.wav", "audio"),
            ("talk.mp4", "video"),
            ("https://youtube.com/watch?v=x", "video"),
            ("files/abc123", "video"),
        ],
    )
    def test_media_kind_detection(self, value: str, expected: str) -> None:
        assert media_kind_for_input(value) == expected

    def test_markdown_header_matches_kind(self) -> None:
        analysis = VideoAnalysis(url="u", raw_response="body")
        assert format_output_markdown(analysis, "audio").startswith("# Audio Analysis")
        assert format_output_markdown(analysis, "video").startswith("# Video Analysis")

    def test_error_header_matches_kind(self) -> None:
        analysis = VideoAnalysis(url="u", error="boom")
        assert "# Error Processing Audio" in format_output_markdown(analysis, "audio")


class TestTruncation:
    def test_banner_present_when_truncated(self) -> None:
        analysis = VideoAnalysis(url="u", raw_response="body", truncated=True)
        assert "TRUNCATED" in format_output_markdown(analysis)

    def test_no_banner_normally(self) -> None:
        analysis = VideoAnalysis(url="u", raw_response="body")
        assert "TRUNCATED" not in format_output_markdown(analysis)

    def test_json_exposes_truncation(self) -> None:
        analysis = VideoAnalysis(
            url="u", raw_response="body", truncated=True, finish_reason="MAX_TOKENS"
        )
        payload = json.loads(format_output_json(analysis))
        assert payload["truncated"] is True
        assert payload["finish_reason"] == "MAX_TOKENS"


class TestTitleExtraction:
    def test_prefers_the_markdown_table_row(self) -> None:
        text = "## Video Information\n\n| Field | Value |\n| **Title** | Real Talk |\n"
        assert extract_title(text) == "Real Talk"

    def test_reads_bold_title_line(self) -> None:
        assert extract_title("**Title**: My Session\n") == "My Session"

    def test_ignores_our_own_heading(self) -> None:
        assert extract_title("# Video Analysis\n\nbody", "fallback") == "fallback"

    def test_falls_back_when_absent(self) -> None:
        assert extract_title("no title here", "fallback") == "fallback"

    def test_does_not_grab_the_word_video(self) -> None:
        """The old regex matched any 'video'/'title' substring anywhere."""
        assert extract_title("This video is about ducks.", "fb") == "fb"


class TestYouTubeParsing:
    def test_json_decode_preserves_unicode(self) -> None:
        assert _decode_json_string(r"Café 🎵") == "Café 🎵"

    def test_json_decode_handles_newlines(self) -> None:
        assert _decode_json_string(r"a\nb") == "a\nb"

    def test_json_decode_survives_malformed_input(self) -> None:
        assert _decode_json_string("bad \\q escape") == "bad \\q escape"

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://www.youtube.com/watch?v=abc123", "abc123"),
            ("https://youtu.be/abc123", "abc123"),
            ("https://www.youtube.com/live/abc123", "abc123"),
            ("https://www.youtube.com/shorts/abc123", "abc123"),
            ("https://www.youtube.com/embed/abc123", "abc123"),
            ("https://m.youtube.com/watch?app=desktop&v=abc123", "abc123"),
        ],
    )
    def test_video_id_extraction(self, url: str, expected: str) -> None:
        assert extract_video_id(url) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/live/abc123",
            "https://m.youtube.com/watch?v=abc123",
            "https://www.youtube.com/watch?app=desktop&v=abc123",
        ],
    )
    def test_live_and_mobile_urls_normalize(self, url: str) -> None:
        assert validate_youtube_url(url) == "https://www.youtube.com/watch?v=abc123"

    def test_duration_from_page(self) -> None:
        with patch(
            "youtube_gemini_processor.cli._fetch_youtube_page",
            return_value='{"lengthSeconds":"3723"}',
        ):
            assert fetch_youtube_duration("https://youtube.com/watch?v=x") == "01:02:03"

    def test_duration_absent_returns_none(self) -> None:
        with patch(
            "youtube_gemini_processor.cli._fetch_youtube_page", return_value="{}"
        ):
            assert fetch_youtube_duration("https://youtube.com/watch?v=x") is None


class TestDurationLine:
    def test_youtube_duration_is_injected(self) -> None:
        with patch(
            "youtube_gemini_processor.cli.fetch_youtube_duration",
            return_value="01:00:00",
        ):
            line = build_duration_line("https://youtube.com/watch?v=x")
        assert "01:00:00" in line

    def test_clip_describes_the_excerpt_not_the_source(self) -> None:
        """Announcing the full duration would ask for time the model never saw."""
        line = build_duration_line("v.mp4", clip_start="60s", clip_end="180s")
        assert "00:02:00" in line
        assert "excerpt" in line

    def test_zero_length_clip_yields_nothing(self) -> None:
        assert build_duration_line("v.mp4", clip_start="60s", clip_end="60s") == ""

    def test_unknown_duration_yields_nothing(self) -> None:
        with patch(
            "youtube_gemini_processor.cli.fetch_youtube_duration", return_value=None
        ):
            assert build_duration_line("https://youtube.com/watch?v=x") == ""


class TestExitCodes:
    """Exiting 0 on failure made every failure look like a success."""

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_failure_exits_one(self, mock_client: MagicMock) -> None:
        client = MagicMock()
        client.vertexai = False
        client.models.generate_content.side_effect = Exception("API down")
        mock_client.return_value = client

        result = CliRunner().invoke(
            main, ["https://youtube.com/watch?v=x", "--api-key", "k"]
        )
        assert result.exit_code == 1

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_success_exits_zero(self, mock_client: MagicMock) -> None:
        client = MagicMock()
        client.vertexai = False
        response = MagicMock()
        response.text = "# Result\n\nbody"
        response.usage_metadata = _usage_meta(prompt=10, candidates=5)
        response.candidates = []
        client.models.generate_content.return_value = response
        mock_client.return_value = client

        result = CliRunner().invoke(
            main, ["https://youtube.com/watch?v=x", "--api-key", "k"]
        )
        assert result.exit_code == 0


class TestCliFlags:
    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_timestamp_offset_reaches_the_prompt(self, mock_client: MagicMock) -> None:
        client = MagicMock()
        client.vertexai = False
        response = MagicMock()
        response.text = "body"
        response.usage_metadata = _usage_meta()
        response.candidates = []
        client.models.generate_content.return_value = response
        mock_client.return_value = client

        result = CliRunner().invoke(
            main,
            [
                "https://youtube.com/watch?v=x",
                "--api-key",
                "k",
                "--timestamp-offset",
                "10:00",
            ],
        )
        assert result.exit_code == 0
        sent = client.models.generate_content.call_args.kwargs["contents"][0]
        assert "00:10:00" in sent.parts[1].text

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_thinking_level_reaches_the_config(self, mock_client: MagicMock) -> None:
        client = MagicMock()
        client.vertexai = False
        response = MagicMock()
        response.text = "body"
        response.usage_metadata = _usage_meta()
        response.candidates = []
        client.models.generate_content.return_value = response
        mock_client.return_value = client

        result = CliRunner().invoke(
            main,
            [
                "https://youtube.com/watch?v=x",
                "--api-key",
                "k",
                "--thinking-level",
                "low",
            ],
        )
        assert result.exit_code == 0
        config = client.models.generate_content.call_args.kwargs["config"]
        assert config.thinking_config.thinking_level == "LOW"

    @patch("youtube_gemini_processor.cli.get_gemini_client")
    def test_unknown_model_warns_but_runs(self, mock_client: MagicMock) -> None:
        client = MagicMock()
        client.vertexai = False
        response = MagicMock()
        response.text = "body"
        response.usage_metadata = _usage_meta()
        response.candidates = []
        client.models.generate_content.return_value = response
        mock_client.return_value = client

        result = CliRunner().invoke(
            main,
            ["https://youtube.com/watch?v=x", "--api-key", "k", "--model", "future-1"],
        )
        assert result.exit_code == 0
        assert "no entry in the pricing table" in result.output


class TestUsageStatsDefaults:
    def test_new_fields_default_safely(self) -> None:
        stats = UsageStats()
        assert stats.thoughts_tokens == 0
        assert stats.audio_input_tokens == 0
        assert stats.cached_tokens == 0
        assert stats.pricing_known is True
