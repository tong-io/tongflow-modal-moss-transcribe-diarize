"""Modal deploy entry for MOSS-Transcribe-Diarize 0.9B (OpenMOSS, Apache-2.0).

An end-to-end model for long-form multi-speaker transcription: it emits
timestamps and anonymous speaker labels in a single pass, rather than stitching
an ASR pass to a separate diarization pipeline. 50+ languages, recordings up to
~90 minutes, and hotword prompting. First place in the 2nd MLC-SLM Challenge at
INTERSPEECH 2026 across 14 languages.

Three slots:

- ``separate_speaker``  — **the reason this plugin exists.** The slot had no
  official implementation. The model only produces a labelled transcript, so
  this cuts the source audio on the diarized spans and concatenates each
  speaker's turns into one track per speaker, returned in label order.
- ``transcribe-timestamp`` — the model's native output shape. Each segment's
  text is prefixed with its speaker label so the diarization survives a slot
  whose ABI has no speaker field.
- ``transcribe`` — the same pass, flattened to plain text.

Whisper keeps the default for the two transcribe slots; pick this plugin when
the recording has several speakers.

The model's canonical output is a single line of
``[start][Sxx]text[end]`` triples, e.g.::

    [0.48][S01]Welcome everyone[1.66][12.26][S02]The pipeline is ready[13.81]

Env knobs (all optional). Read when ``modal deploy`` runs and baked into the
image, because Modal re-imports this module in the container where the deploy
shell's environment is absent — re-deploy after changing:
  MOSS_GPU              Modal GPU, default "L4" ($0.80/h). The model is 0.9B,
                        so the small card is the right one; raise it only for
                        very long recordings.
  MOSS_MAX_NEW_TOKENS   decoder cap, default 8192. Long multi-speaker audio
                        needs headroom or the transcript stops mid-way; the
                        ABI's max_new_tokens overrides it per request.

Deploy:           modal deploy deploy.py
Download weights: modal run download.py::download
"""

from __future__ import annotations

import os
from pathlib import Path

import modal
from tongflow import deploy
from tongflow.models.separate_speaker import SeparateSpeakerInput, SeparateSpeakerOutput
from tongflow.models.transcribe import TranscribeInput, TranscribeOutput
from tongflow.models.transcribe_timestamp import (
    TranscribeTimestampInput,
    TranscribeTimestampOutput,
    TranscribeTimestampOutputRootTimeStampsItem,
)
from tongflow.node_slots import NodeSlots
from tongflow.protocol import asset, prompt_media_to_bytes
from tongflow.slots import node_slot

# separate_speaker had no official implementation before this plugin; the two
# transcribe slots stay with Whisper.
TONGFLOW_DEFAULT_SLOTS = ["separate_speaker"]

REPO_ID = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
MODEL_DIR = f"/models/{REPO_ID}"

GPU = (os.environ.get("MOSS_GPU") or "L4").strip()
MAX_NEW_TOKENS = int(os.environ.get("MOSS_MAX_NEW_TOKENS") or 8192)

# Snapshot of the MOSS_* knobs as the deploy shell saw them, baked into the
# image so the constants above resolve identically inside the container.
_DEPLOY_ENV = {k: v for k, v in os.environ.items() if k.startswith("MOSS_")}

volume = modal.Volume.from_name("models", create_if_missing=True)

app = modal.App(Path(__file__).resolve().parent.name)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==2.7.1",
        "torchaudio==2.7.1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers>=4.51",
        "accelerate",
        "librosa",
        "soundfile",
        # Helper package: chat-template message construction, audio loading and
        # transcript parsing. The weights and the remote code come from HF.
        "git+https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git",
    )
    .pip_install("tongflow==0.3.2", "fastapi[standard]")
    .env({"HF_HOME": "/models/hf", **_DEPLOY_ENV})
)

with image.imports():
    import re
    import subprocess
    import tempfile

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

# One [start][Sxx]text[end] triple. Parsed here rather than via the helper's
# parse_transcript so a malformed tail degrades to "skip that span" instead of
# failing the whole request.
_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\](?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
    re.S,
)


def _parse(transcript: str) -> list[dict]:
    """[{start, end, speaker, text}] for every well-formed span, in order."""
    out: list[dict] = []
    for m in _SEGMENT_RE.finditer(transcript or ""):
        start, end = float(m.group("start")), float(m.group("end"))
        text = m.group("text").strip()
        if end > start and text:
            out.append(
                {"start": start, "end": end, "speaker": m.group("speaker"), "text": text}
            )
    return out


def _prompt_with_hotwords(context: str) -> str | None:
    """Default instruction + a hotword hint, or None to use the default as-is.

    ``build_transcription_messages(prompt=...)`` *replaces* the instruction
    rather than extending it, so passing the hint alone drops the "transcribe
    with timestamps and speaker labels" part and the model returns text that
    parses to nothing. The hint has to be appended to DEFAULT_PROMPT."""
    context = (context or "").strip()
    if not context:
        return None
    from moss_transcribe_diarize.inference_utils import DEFAULT_PROMPT

    return f"{DEFAULT_PROMPT}热词提示：{context}"


@deploy
@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/models": volume},
    timeout=3600,
    scaledown_window=2,
)
class Inference:
    @modal.enter()
    def _boot(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                trust_remote_code=True,
                dtype="auto",
                attn_implementation="sdpa",
            )
            .to(dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

    # ── core pass ──────────────────────────────────────────────────────

    def _transcribe(self, audio_bytes: bytes, filename: str, context: str,
                    max_new_tokens: int) -> list[dict]:
        """Run one diarized pass and return the parsed segments."""
        from moss_transcribe_diarize.inference_utils import (
            build_transcription_messages,
            generate_transcription,
        )

        suffix = Path(filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            src = f.name
        try:
            prompt = _prompt_with_hotwords(context)
            messages = (
                build_transcription_messages(src, prompt=prompt)
                if prompt
                else build_transcription_messages(src)
            )
            result = generate_transcription(
                self.model,
                self.processor,
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                device=self.device,
                dtype=self.dtype,
            )
        finally:
            os.unlink(src)
        return _parse(result["text"])

    @staticmethod
    def _cap(value: object) -> int:
        try:
            return int(value) if value is not None else MAX_NEW_TOKENS
        except (TypeError, ValueError):
            return MAX_NEW_TOKENS

    # ── slots ──────────────────────────────────────────────────────────

    @modal.method()
    @node_slot(NodeSlots.TRANSCRIBE)
    def transcribe(self, input: TranscribeInput) -> TranscribeOutput:
        try:
            segments = self._transcribe(
                prompt_media_to_bytes(input.audio),
                input.audio.filename or "audio.wav",
                input.context or "",
                self._cap(input.max_new_tokens),
            )
        except Exception as e:
            return TranscribeOutput(success=False, error=str(e))
        text = "\n".join(f"[{s['speaker']}] {s['text']}" for s in segments)
        return TranscribeOutput(success=True, text=text)

    @modal.method()
    @node_slot(NodeSlots.TRANSCRIBE_TIMESTAMP)
    def transcribe_timestamp(
        self, input: TranscribeTimestampInput
    ) -> TranscribeTimestampOutput:
        try:
            segments = self._transcribe(
                prompt_media_to_bytes(input.audio),
                input.audio.filename or "audio.wav",
                input.context or "",
                self._cap(input.max_new_tokens),
            )
        except Exception as e:
            return TranscribeTimestampOutput(success=False, error=str(e))
        # The slot has no speaker field, so carry the label in the text.
        stamps = [
            TranscribeTimestampOutputRootTimeStampsItem(
                start=s["start"], end=s["end"], text=f"[{s['speaker']}] {s['text']}"
            )
            for s in segments
        ]
        return TranscribeTimestampOutput(
            success=True,
            text="\n".join(f"[{s['speaker']}] {s['text']}" for s in segments),
            time_stamps=stamps,
        )

    @modal.method()
    @node_slot(NodeSlots.SEPARATE_SPEAKER)
    def separate_speaker(self, input: SeparateSpeakerInput) -> SeparateSpeakerOutput:
        """One audio track per speaker, cut from the diarized spans.

        The model labels turns but does not un-mix them, so each track is that
        speaker's own turns concatenated — silence elsewhere is dropped rather
        than preserved, which keeps the tracks short and directly listenable.
        """
        try:
            raw = prompt_media_to_bytes(input.audio)
            segments = self._transcribe(
                raw, input.audio.filename or "audio.wav", "", MAX_NEW_TOKENS
            )
        except Exception as e:
            return SeparateSpeakerOutput(success=False, error=str(e))
        if not segments:
            return SeparateSpeakerOutput(
                success=False, error="No speech segments were detected"
            )

        by_speaker: dict[str, list[dict]] = {}
        for s in segments:
            by_speaker.setdefault(s["speaker"], []).append(s)

        suffix = Path(input.audio.filename or "audio.wav").suffix or ".wav"
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, f"src{suffix}")
            with open(src, "wb") as f:
                f.write(raw)
            tracks: list = []
            for speaker in sorted(by_speaker):
                # One trim per turn, concatenated in the filter graph: a single
                # ffmpeg pass per speaker, no intermediate files.
                spans = by_speaker[speaker]
                parts = "".join(
                    f"[0:a]atrim=start={s['start']}:end={s['end']},"
                    f"asetpts=PTS-STARTPTS[a{i}];"
                    for i, s in enumerate(spans)
                )
                inputs = "".join(f"[a{i}]" for i in range(len(spans)))
                graph = f"{parts}{inputs}concat=n={len(spans)}:v=0:a=1[out]"
                dst = os.path.join(tmp, f"{speaker}.wav")
                proc = subprocess.run(
                    ["ffmpeg", "-y", "-i", src, "-filter_complex", graph,
                     "-map", "[out]", dst],
                    capture_output=True, text=True,
                )
                if proc.returncode != 0 or not os.path.isfile(dst):
                    return SeparateSpeakerOutput(
                        success=False,
                        error=f"ffmpeg failed for {speaker}: {proc.stderr[-500:]}",
                    )
                with open(dst, "rb") as fh:
                    tracks.append(
                        asset(fh.read(), mime="audio/wav", filename=f"{speaker}.wav")
                    )
        return SeparateSpeakerOutput(success=True, outputKeys=tracks)

    @modal.fastapi_endpoint(method="GET", label=f"{Path(__file__).resolve().parent.name}-serve")
    def serve(self, taskId: str = "", token: str = "", origin: str = ""):
        from fastapi.responses import StreamingResponse
        from tongflow import serve_stream_from_spec

        return StreamingResponse(
            serve_stream_from_spec(
                origin, taskId, token, __file__,
                invoke=lambda m, inp: getattr(self, m).local(inp),
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Access-Control-Allow-Origin": "*"},
        )
