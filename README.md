# tongflow-modal-moss-transcribe-diarize

Self-hosted [MOSS-Transcribe-Diarize 0.9B](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
(OpenMOSS / MOSI.AI, **Apache-2.0**) for [TongFlow](https://github.com/tong-io/tongflow),
on [Modal](https://modal.com).

One end-to-end pass turns long, messy, multi-speaker audio into a
speaker-labelled transcript with timestamps — no separate ASR and diarization
stages to stitch together. 50+ languages, recordings up to ~90 minutes, and
hotword prompting for domain terms. It won the 2nd MLC-SLM Challenge at
INTERSPEECH 2026 across 14 languages.

At 0.9B it is small enough for an L4, which is what makes running it locally
worthwhile at all.

## Slots

| Slot | Output |
|---|---|
| `separate_speaker` **(default)** | one audio track per speaker |
| `transcribe-timestamp` | `[{start, end, text}]` with the speaker label carried in `text` |
| `transcribe` | plain text, one labelled line per turn |

Whisper keeps the default for the two transcribe slots — pick this plugin from
the node's plugin picker when the recording has **several speakers**. It is the
first official implementation of `separate_speaker`.

**About `separate_speaker`:** the model labels turns, it does not un-mix
overlapping voices. Each returned track is that speaker's own turns cut from
the source and concatenated, so the tracks are short and directly listenable
but do **not** preserve the original timeline, and audio where two people talk
over each other lands in both tracks. For true source separation (vocals,
drums, an arbitrary described sound) use the ACE-Step or SAM-Audio plugins.

Speaker labels (`[S01]`, `[S02]`, …) are relative to one recording. They are
not identities and do not carry across separate runs.

**Hotwords:** the ABI's `context` field is forwarded as the model's hotword
hint, which is the documented way to bias it toward names, jargon or product
terms. Pass a comma-separated list.

## Model / GPU

Weights land on the shared `models` Modal volume at
`/models/OpenMOSS-Team/MOSS-Transcribe-Diarize` (~2 GB — the full snapshot,
since the model loads its own remote code with `trust_remote_code=True`).

Default GPU is **L4** ($0.80/h on Modal). Env knobs are read when
`modal deploy` runs and **baked into the image**, because Modal re-imports the
module inside the container where the deploy shell's environment does not
exist:

| Env | Default | Purpose |
|---|---|---|
| `MOSS_GPU` | `L4` | Raise only for very long recordings. |
| `MOSS_MAX_NEW_TOKENS` | `8192` | Decoder cap per pass. Long multi-speaker audio needs headroom or the transcript stops mid-way; the ABI's `max_new_tokens` overrides it per request. |

## Test runbook (run these yourself — every step below bills Modal)

Use the venv modal client (`sdk/.venv/bin/modal` from the tongflow repo) with
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` exported, from this plugin directory.

1. **Download weights** (~2 GB, CPU-only, one-off): `modal run download.py::download`
2. **Deploy**: `modal deploy deploy.py`
3. **Two-speaker smoke test**: transcribe a short interview or podcast clip.
   Confirm the turns alternate between `[S01]` and `[S02]` at the right
   moments, and that the timestamps line up with the audio.
4. **`separate_speaker`**: same clip. Expect one WAV per speaker; play each and
   confirm it contains only that voice.
5. **Hotwords**: re-run with `context` set to a name the model got wrong in
   step 3, and check it comes out right.
6. **Length**: try something over ~10 minutes. If the transcript stops early,
   raise `MOSS_MAX_NEW_TOKENS` — that is the cap, not a model limit.

## Notes

- The model's canonical output is one line of `[start][Sxx]text[end]` triples.
  `deploy.py` parses it with a tolerant regex rather than the helper package's
  `parse_transcript`, so a malformed tail drops that one span instead of
  failing the whole request.
- `separate_speaker` builds each track in a single ffmpeg pass per speaker
  (one `atrim` per turn, concatenated in the filter graph) — no intermediate
  files.
- Serving via SGLang Omni or vLLM is possible upstream and would batch better,
  but this plugin runs the plain transformers path: one request at a time is
  what a workflow node does, and it keeps the image small.
