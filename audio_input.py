import contextlib
import os
import subprocess
import threading
import time

import numpy as np
import sounddevice as sd


PULSE_SOURCE_ENV = "JARVIS_PULSE_SOURCE"
DEFAULT_BLUETOOTH_SOURCE = "bluez_input.E8_6B_EA_31_C3_66.0"
PULSE_ENV = {
    "XDG_RUNTIME_DIR": "/run/user/1000",
    "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
}


def _pulse_env():
    env = os.environ.copy()
    for key, value in PULSE_ENV.items():
        env.setdefault(key, value)
    return env


def get_sounddevice_input_device():
    devices = sd.query_devices()
    for index, device in enumerate(devices):
        if device.get("max_input_channels", 0) > 0:
            return index
    return None


def get_pulse_source():
    configured = os.getenv(PULSE_SOURCE_ENV)
    if configured:
        return configured

    try:
        result = subprocess.run(
            ["pactl", "get-default-source"],
            check=True,
            capture_output=True,
            text=True,
            env=_pulse_env(),
        )
        source = result.stdout.strip()
        if source:
            return source
    except Exception:
        pass

    return DEFAULT_BLUETOOTH_SOURCE


def describe_audio_inputs():
    lines = []
    with contextlib.suppress(Exception):
        for index, device in enumerate(sd.query_devices()):
            lines.append(
                f"sounddevice {index}: {device['name']} "
                f"({device['max_input_channels']} in, {device['max_output_channels']} out)"
            )

    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            check=True,
            capture_output=True,
            text=True,
            env=_pulse_env(),
        )
        for line in result.stdout.splitlines():
            lines.append(f"pulse {line}")
    except Exception as exc:
        lines.append(f"pulse unavailable: {exc}")

    return "\n".join(lines) or "none"


class PulseInputStream:
    def __init__(
        self,
        *,
        samplerate,
        channels,
        dtype,
        blocksize,
        callback=None,
        source=None,
    ):
        if dtype != "int16":
            raise ValueError("PulseInputStream only supports int16 PCM")
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.blocksize = blocksize
        self.callback = callback
        self.source = source or get_pulse_source()
        self._closed = threading.Event()
        self._thread = None
        self._error = None
        self._process = None

    def __enter__(self):
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "pulse",
                "-i",
                self.source,
                "-ac",
                str(self.channels),
                "-ar",
                str(self.samplerate),
                "-f",
                "s16le",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_pulse_env(),
        )
        if self.callback:
            self._thread = threading.Thread(target=self._callback_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        self._closed.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self._process.wait(timeout=2)
            if self._process.poll() is None:
                self._process.kill()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def read(self, frames):
        needed = frames * self.channels * 2
        data = self._read_exactly(needed)
        samples = np.frombuffer(data, dtype=np.int16).reshape(-1, self.channels)
        return samples, None

    def _read_exactly(self, needed):
        if not self._process or not self._process.stdout:
            raise RuntimeError("Pulse input process is not running")

        chunks = []
        remaining = needed
        while remaining > 0 and not self._closed.is_set():
            chunk = self._process.stdout.read(remaining)
            if chunk:
                chunks.append(chunk)
                remaining -= len(chunk)
                continue

            stderr = b""
            if self._process.stderr:
                with contextlib.suppress(Exception):
                    stderr = self._process.stderr.read()
            message = stderr.decode(errors="replace").strip()
            raise RuntimeError(
                f"Pulse microphone stream ended unexpectedly: {message or 'no details'}"
            )

        if self._closed.is_set():
            raise RuntimeError("Pulse input stream is closed")

        return b"".join(chunks)

    def _callback_loop(self):
        needed = self.blocksize * self.channels * 2
        while not self._closed.is_set():
            try:
                data = self._read_exactly(needed)
                samples = np.frombuffer(data, dtype=np.int16).reshape(
                    -1, self.channels
                )
                self.callback(samples, len(samples), None, None)
            except Exception as exc:
                self._error = str(exc)
                time.sleep(0.25)


def open_input_stream(*, samplerate, channels, dtype, blocksize, callback=None):
    device = get_sounddevice_input_device()
    if device is not None:
        return sd.InputStream(
            device=device,
            samplerate=samplerate,
            channels=channels,
            dtype=dtype,
            blocksize=blocksize,
            callback=callback,
        )

    return PulseInputStream(
        samplerate=samplerate,
        channels=channels,
        dtype=dtype,
        blocksize=blocksize,
        callback=callback,
    )
