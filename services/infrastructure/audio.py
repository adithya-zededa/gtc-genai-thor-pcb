"""Infrastructure service for audio playback.

Provides platform-agnostic audio playback using available system players.
Can be reused across domains (retail TTS, alert sounds, notifications, etc.).

Supports configurable audio device via SPEAKER_DEVICE environment variable.
Supports volume control via SPEAKER_VOLUME environment variable (0-100).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional, List, Tuple

from core.logging import get_logger

logger = get_logger(__name__)


class AudioPlayer:
    """Platform-agnostic audio player using system CLI tools.
    
    Supports device configuration via SPEAKER_DEVICE environment variable.
    Supports volume control via SPEAKER_VOLUME environment variable (0-100).
    For example:
    - SPEAKER_DEVICE="hw:0,0" for ALSA device
    - SPEAKER_DEVICE="plughw:0,0" for ALSA with automatic conversion
    - SPEAKER_VOLUME="85" for 85% volume
    """

    def __init__(self):
        """Initialize the AudioPlayer with device and volume configuration."""
        self._speaker_device = os.getenv("SPEAKER_DEVICE", "")
        self._volume = self._parse_volume(os.getenv("SPEAKER_VOLUME", "85"))
        if self._speaker_device:
            logger.info("AudioPlayer configured with device: %s, volume: %d%%", 
                       self._speaker_device, self._volume)
        else:
            logger.info("AudioPlayer configured with volume: %d%%", self._volume)

    def _parse_volume(self, volume_str: str) -> int:
        """Parse and validate volume from string.
        
        Parameters
        ----------
        volume_str : str
            Volume value as string (0-100).
            
        Returns
        -------
        int
            Valid volume value clamped between 0 and 100.
        """
        try:
            volume = int(volume_str)
            # Clamp volume between 0 and 100
            return max(0, min(100, volume))
        except (ValueError, TypeError):
            logger.warning("Invalid volume value '%s', using default 85", volume_str)
            return 85

    def _get_player_configs(self) -> List[Tuple[str, List[str]]]:
        """Get list of (command, extra_args) for available players.
        
        Returns device-specific and volume arguments when configured.
        Prioritizes players that work in headless/container environments.
        
        Returns
        -------
        List[Tuple[str, List[str]]]
            List of (player_command, args) tuples.
        """
        configs = []
        
        # mpv - supports audio device selection and volume (works headless)
        mpv_args = ["--no-video", "--really-quiet"]
        # mpv volume is 0-100
        mpv_args.extend(["--volume", str(self._volume)])
        if self._speaker_device:
            mpv_args.extend(["--audio-device", f"alsa/{self._speaker_device}"])
        configs.append(("mpv", mpv_args))
        
        # ffmpeg - reliable audio player for containers (use audio filter for volume)
        # Works well in headless environments
        ffmpeg_args = ["-loglevel", "error", "-i"]  # -i will be followed by the file path
        # Add volume filter (range 0.0-10.0, where 1.0 is 100%)
        volume_factor = self._volume / 100.0
        ffmpeg_args.extend(["-af", f"volume={volume_factor}"])
        if self._speaker_device:
            # Use ALSA output with specific device
            ffmpeg_args.extend(["-f", "alsa", self._speaker_device])
        else:
            # Use default ALSA output
            ffmpeg_args.extend(["-f", "alsa", "default"])
        configs.append(("ffmpeg", ffmpeg_args))
        
        # aplay - ALSA player with direct device support (WAV only)
        # Note: aplay doesn't play MP3, so this is only for WAV files
        aplay_args = []
        if self._speaker_device:
            aplay_args.extend(["-D", self._speaker_device])
        configs.append(("aplay", aplay_args))
        
        # afplay - macOS (no device selection via CLI)
        # afplay volume is 0.0-1.0
        afplay_volume = self._volume / 100.0
        afplay_args = ["-v", str(afplay_volume)]
        configs.append(("afplay", afplay_args))
        
        return configs

    def _set_alsa_volume(self) -> None:
        """Set ALSA volume using amixer if available.
        
        This is particularly useful for aplay which doesn't have built-in
        volume control. amixer sets the volume at the system level.
        """
        if not shutil.which("amixer"):
            return
            
        try:
            # Try to set Master volume
            cmd = ["amixer", "set", "Master", f"{self._volume}%"]
            if self._speaker_device:
                # Specify card from device (e.g., "hw:2,0" -> card 2)
                if self._speaker_device.startswith("hw:") or self._speaker_device.startswith("plughw:"):
                    card_part = self._speaker_device.split(":")[1].split(",")[0]
                    cmd = ["amixer", "-c", card_part, "set", "Master", f"{self._volume}%"]
            
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
            logger.debug("ALSA volume set to %d%% via amixer", self._volume)
        except Exception as exc:
            logger.debug("Could not set ALSA volume via amixer: %s", exc)

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
        player_configs = self._get_player_configs()
        
        for player, extra_args in player_configs:
            if shutil.which(player):
                # Set ALSA volume before using aplay (which doesn't have volume control)
                if player == "aplay":
                    self._set_alsa_volume()
                
                # Build command based on player
                if player == "ffmpeg":
                    # ffmpeg needs input file after -i flag
                    # extra_args already contains ["-loglevel", "error", "-i"]
                    cmd = [player] + extra_args[:3] + [audio_path] + extra_args[3:]
                else:
                    # Other players take file path at the end
                    cmd = [player] + extra_args + [audio_path]
                
                try:
                    subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    device_info = f" (device: {self._speaker_device})" if self._speaker_device else ""
                    logger.info("Audio playback started via %s%s at %d%% volume", 
                               player, device_info, self._volume)
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
        # Create temporary instance to get player configs
        temp_instance = cls()
        player_configs = temp_instance._get_player_configs()
        return any(shutil.which(player) for player, _ in player_configs
            True if at least one player is found.
        """
        return any(shutil.which(player) for player, _ in cls._PLAYERS)
