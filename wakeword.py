import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model
from audio_input import describe_audio_inputs, open_input_stream
try:
    from openwakeword.utils import download_models
except ImportError:
    download_models = None


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280
THRESHOLD = 0.5
DEBOUNCE_SECONDS = 2.0
CUSTOM_MODEL = BASE_DIR / "models" / "ahh_niu__ahh_niu.onnx"
FALLBACK_MODEL = "hey_jarvis"


def build_model(custom_model: str | Path = CUSTOM_MODEL):
    custom_model = Path(custom_model)
    if custom_model.exists():
        return _load_model([str(custom_model)])

    print(
        f"{custom_model.name} not found; using built-in openWakeWord model "
        f"'{FALLBACK_MODEL}'."
    )
    available_models = getattr(openwakeword, "MODELS", None) or getattr(
        openwakeword, "models", {}
    )
    if FALLBACK_MODEL not in available_models:
        raise RuntimeError(
            f"Built-in openWakeWord model '{FALLBACK_MODEL}' is not available."
        )

    fallback_model_path = available_models[FALLBACK_MODEL]["model_path"].replace(
        ".tflite", ".onnx"
    )
    if not os.path.exists(fallback_model_path) and download_models:
        download_models([FALLBACK_MODEL])
    if os.path.exists(fallback_model_path):
        return _load_model([fallback_model_path])

    return _load_model([FALLBACK_MODEL])


def _load_model(model_paths: list[str]):
    try:
        return Model(wakeword_model_paths=model_paths)
    except TypeError:
        return Model(wakeword_models=model_paths, inference_framework="onnx")


class WakeWordDetector:
    def __init__(
        self,
        threshold: float = THRESHOLD,
        debounce_seconds: float = DEBOUNCE_SECONDS,
        frame_samples: int = FRAME_SAMPLES,
        custom_model: str | Path = CUSTOM_MODEL,
        model: Model | None = None,
    ):
        self.model = model or build_model(custom_model)
        self.threshold = threshold
        self.debounce_seconds = debounce_seconds
        self.frame_samples = frame_samples
        self._buffer = np.empty(0, dtype=np.int16)
        self._last_detection = 0.0
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self._buffer = np.empty(0, dtype=np.int16)
            self._last_detection = 0.0
            reset_model = getattr(self.model, "reset", None)
            if callable(reset_model):
                reset_model()

    def process_bytes(self, data: bytes):
        if not data:
            return None
        return self.process_audio(np.frombuffer(data, dtype=np.int16))

    def process_audio(self, audio):
        with self._lock:
            samples = np.asarray(audio, dtype=np.int16).reshape(-1)
            if samples.size == 0:
                return None

            self._buffer = np.concatenate((self._buffer, samples))
            while self._buffer.size >= self.frame_samples:
                frame = self._buffer[: self.frame_samples].copy()
                self._buffer = self._buffer[self.frame_samples :]
                scores = self.model.predict(frame)
                now = time.monotonic()

                for name, confidence in scores.items():
                    if (
                        confidence >= self.threshold
                        and now - self._last_detection >= self.debounce_seconds
                    ):
                        self._last_detection = now
                        self._buffer = np.empty(0, dtype=np.int16)
                        return name, float(confidence)

        return None


def main():
    detector = WakeWordDetector()

    print("Listening for wake word. Press Ctrl+C to stop.")
    with open_input_stream(
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
    ) as stream:
        while True:
            frame, _ = stream.read(FRAME_SAMPLES)
            detection = detector.process_audio(np.squeeze(frame))
            if detection:
                name, confidence = detection
                print(f"Detected {name}! ({confidence:.2f})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped.")
    except (RuntimeError, sd.PortAudioError) as exc:
        print(f"wakeword error: {exc}\nDetected inputs:\n{describe_audio_inputs()}", file=sys.stderr)
        sys.exit(1)
