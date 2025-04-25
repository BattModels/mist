import logging
import time


class RateLimitedAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None, min_interval=5.0):
        super().__init__(logger, extra or {})
        self.min_interval = min_interval
        self._last_log_time = {}

    def _rate_limited(self, key):
        now = time.time()
        # If log is a dict, we use the keys as the log key
        key = key if not isinstance(key, dict) else "\0".join(key.keys())
        last_time = self._last_log_time.get(key, None)
        if last_time is None or now - last_time >= self.min_interval:
            self._last_log_time[key] = now
            return False
        return True

    def log(self, level, msg, *args, key=None, **kwargs):
        log_key = key or msg  # fallback to message string if no explicit key
        if not self._rate_limited(log_key):
            super().log(level, msg, *args, **kwargs)
