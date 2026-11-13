# Changelog

All notable changes to PrivateVoice_Ahee52 are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/), versions use `MAJOR.MINOR.PATCH`.

## [1.0.5] — 2026-11-13

### Fixed
- Training failed at `1-get-text.py`: `huggingface_hub` 2.4 removed `HfFolder`, which gradio 4.44
  still imports. Runtime now pins `huggingface_hub==0.26.2`.
- Settings could not open: `Config.root` does not exist; the whisper model directory is now
  resolved from `gpt_sovits`.
- Hotkey could not be captured back to `Shift+A` while the global hotkey was active — the hotkey
  is suspended while Settings is open.
- LLM provider dialog crashed with `NameError` (`tz(超时 (ms))`); add/edit works again.
- Whisper returned “didn't catch that” on valid speech — retries without VAD and relaxes the
  speech thresholds.
- English ASR output had no spaces or punctuation — results are normalised (spacing, punctuation,
  capitalisation, final period).
- English input produced Chinese replies — the language directive moved to the end of the system
  prompt and is repeated in the user turn; replies follow zh/en/ja/ko.
- Switching the ASR model now restarts the app automatically (1.2 s) instead of failing quietly.
- Settings and LLM pages are fully translated in English mode.
- Bundled scripts renamed to English: `Launch.bat`, `SelfTest.bat`, `MicCheck.bat`, `README.txt`.
- chat-wave icon applied to app, tray, installer and docs assets.

## [1.0.4] — 2026-11-13

### Fixed
- Rolled back to the 1.0.3 baseline after 1.1.x regressions.
- Startup crash caused by a missing `QIcon` import (1.1.x only) — not present on the baseline,
  plus a global exception hook that logs crashes and shows a dialog.
- ASR reload on settings change; model path uses `models_cache`, not `models`.
- LLM URL normalisation (`/v1`, duplicate `chat/completions`), legacy `llm.base_url` fallback,
  optional API key.
- System default output device resolved on every stream open.
- Self-test uses the default character's `reference.wav`; output is English only.

## [1.0.3] — 2026-11-05

### Added
- Split README: `README.md` (English) and `README.zh.md` (Chinese).
- MIT `LICENSE`.

### Changed
- Repository moved to `C:\Local Repositories\PrivateVoice`; installer targets
  `%ProgramFiles%\PrivateVoice_Ahee52`.

## [1.0.0] — 2026-11-05

### Added
- First stable build: bilingual UI, local ASR/LLM/TTS pipeline, character training with QC,
  self-test, online lookup (Bing / Baidu / Wikipedia).
- Installer split into <2 GB slices for GitHub Releases.
