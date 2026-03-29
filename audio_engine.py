from __future__ import annotations

from pathlib import Path

import pygame


class AudioEngine:
    def __init__(self, audio_path: str = "assets/mlg_audio.mp3") -> None:
        self.audio_path = Path(audio_path)
        self.sound: pygame.mixer.Sound | None = None
        self.audio_enabled = False

        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.audio_enabled = True
        except pygame.error as exc:
            print(f"[audio] mixer init failed ({exc}) — audio disabled")
            return

        self._load()

    def _load(self) -> None:
        if not self.audio_enabled:
            return

        resolved_path = self._resolve_audio_path()
        if resolved_path is None:
            print("[audio] no audio file found in assets — audio disabled")
            return

        try:
            self.sound = pygame.mixer.Sound(str(resolved_path))
            self.audio_path = resolved_path
        except FileNotFoundError:
            print("[audio] mlg_audio.mp3 not found — audio disabled")
        except pygame.error as exc:
            print(f"[audio] failed to load audio ({exc}) — audio disabled")

    def _resolve_audio_path(self) -> Path | None:
        if self.audio_path.exists():
            return self.audio_path

        assets_dir = self.audio_path.parent
        if not assets_dir.exists():
            return None

        for pattern in ("mlg_audio.mp3", "*.mp3", "*.wav", "*.ogg"):
            matches = sorted(assets_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def play(self) -> None:
        if self.sound:
            self.sound.play(loops=-1)

    def stop(self) -> None:
        if self.sound:
            self.sound.stop()
