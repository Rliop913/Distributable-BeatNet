from __future__ import annotations

from pathlib import Path

import numpy as np


class AudioFeatureAdapter:
    """Optional adapter that reuses the original BeatNet audio preprocessing."""

    def __init__(
        self,
        sample_rate: int = 22050,
        win_length_ms: int = 64,
        hop_length_ms: int = 20,
    ) -> None:
        self.sample_rate = sample_rate
        self.win_length = int(win_length_ms * 0.001 * sample_rate)
        self.hop_length = int(hop_length_ms * 0.001 * sample_rate)
        self._processor = None

    def _get_processor(self):
        if self._processor is None:
            from BeatNet.log_spect import LOG_SPECT

            self._processor = LOG_SPECT(
                sample_rate=self.sample_rate,
                win_length=self.win_length,
                hop_size=self.hop_length,
                n_bands=[24],
                mode="online",
            )
        return self._processor

    def load_audio(self, audio_source: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(audio_source, (str, Path)):
            import librosa

            audio, _ = librosa.load(str(audio_source), sr=self.sample_rate)
            return np.asarray(audio, dtype=np.float32)

        audio = np.asarray(audio_source, dtype=np.float32)
        if audio.ndim == 2:
            channel_axis = 0 if audio.shape[0] <= audio.shape[1] else 1
            audio = audio.mean(axis=channel_axis)
        elif audio.ndim != 1:
            raise ValueError(f"Expected 1D or 2D audio input; got {audio.shape}")
        return np.asarray(audio, dtype=np.float32)

    def audio_to_features(self, audio_source: str | Path | np.ndarray) -> np.ndarray:
        audio = self.load_audio(audio_source)
        feats = self._get_processor().process_audio(audio).T
        return np.asarray(feats, dtype=np.float32)
