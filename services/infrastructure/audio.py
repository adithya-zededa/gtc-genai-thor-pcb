"""Infrastructure service for audio playback.

Provides platform-agnostic audio playback using available system players.
Can be reused across domains (retail TTS, alert sounds, notifications, etc.).
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Optional

from core.logging import get_logger

logger = get_logger(__name__)


class AudioPlayer:
    """Platform-agnostic audio player using system CLI tools."""

    # List of (command, extra_args) to try in order
    _PLAYERS = [
        ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
        ("mpv", ["--no-video", "--really-quiet"]),
        ("aplay", []),   # Linux ALSA (wav only — may not work for mp3)
        ("afplay", []),  # macOS
    ]

    def play_file(self, audio_path: str) -> bool:
        """Attempt to play an audio file using available system players.

        This is a best-effort operation. If no player is found, the method
        returns False but the audio file remains available on disk.

        Parameters
        ----------
        audio_path : str
            Path to the audio file to play.

        Returns
        -------
        bool
            True if playback was initiated, False otherwise.
        """
        for player, extra_args in self._PLAYERS:
            if shutil.which(player):
                cmd = [player] + extra_args + [audio_path]
                try:
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    logger.info("Audio playback started via %s", player)
                    return True
                except Exception as exc:
                    logger.debug("Player %s failed: %s", player, exc)

        logger.warning(
            "No audio player found on the system. Audio saved to: %s",
            audio_path,
        )
        return False

    @classmethod
    def is_available(cls) -> bool:
        """Check if any audio player is available on the system.

        Returns
        -------
        bool
            True if at least one player is found.
        """
        return any(shutil.which(player) for player, _ in cls._PLAYERS)
