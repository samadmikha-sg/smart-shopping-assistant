"""Text-to-speech with a cooldown so the same message is not repeated constantly."""
from __future__ import annotations

import time
from typing import Callable


class Speaker:
    def __init__(self, cooldown: float = 4.0, backend: Callable[[str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.cooldown = cooldown
        self._clock = clock
        self._last: dict[str, float] = {}
        self._backend = backend or self._default_backend()

    @staticmethod
    def _default_backend() -> Callable[[str], None]:
        try:
            import pyttsx3  # optional dependency
            engine = pyttsx3.init()

            def speak(text: str) -> None:
                engine.say(text)
                engine.runAndWait()
            return speak
        except Exception:
            return lambda text: print(f"[speech] {text}")

    def say(self, text: str) -> bool:
        """Speak text unless it was spoken within the cooldown window. Returns True if spoken."""
        now = self._clock()
        if text in self._last and now - self._last[text] < self.cooldown:
            return False
        self._last[text] = now
        self._backend(text)
        return True
