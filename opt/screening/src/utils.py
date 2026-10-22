import json
import logging
import shutil
import time
from itertools import takewhile
from pathlib import Path

START_TIME = time.perf_counter()


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


class JSONLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if isinstance(record.msg, dict):
            log_entry = record.msg
        else:
            log_entry = {"message": record.getMessage()}
        log_entry["level"] = record.levelname
        log_entry["global_rank"] = getattr(record, "global_rank", "unknown")
        log_entry["world_size"] = getattr(record, "world_size", "unknown")
        log_entry["name"] = record.name
        log_entry["elapsed_perf"] = getattr(record, "elapsed_perf", None)
        return json.dumps(log_entry)


class ElapsedTimeFilter(logging.Filter):
    def filter(self, record):
        record.elapsed_perf = time.perf_counter() - START_TIME
        return True


class FabricRankFilter(logging.Filter):
    def __init__(self, fabric):
        super().__init__()
        self.fabric = fabric

    def filter(self, record):
        record.global_rank = self.fabric.global_rank
        record.world_size = self.fabric.world_size
        return True


def configure_logging(fabric, log_file, logger=None):
    logger = logger or logging.getLogger()
    log_file = Path(log_file)
    # Appending from multiple nodes is not atomic on the shared filesystem.
    file_handler = logging.FileHandler(rank_log_path(log_file, fabric.global_rank), mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JSONLineFormatter())

    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.addFilter(FabricRankFilter(fabric))
    logger.addFilter(ElapsedTimeFilter())


def rank_log_path(log_file, rank):
    log_file = Path(log_file)
    return log_file.with_name(f"{log_file.stem}.rank-{rank}{log_file.suffix}")


def merge_rank_logs(log_file, world_size):
    """Publish the trace after every rank has finished writing its own file."""
    log_file = Path(log_file)
    temporary = log_file.with_suffix(log_file.suffix + ".tmp")
    with temporary.open("w") as output:
        for rank in range(world_size):
            with rank_log_path(log_file, rank).open() as source:
                shutil.copyfileobj(source, output)
    temporary.replace(log_file)


def take_for_seconds(iterable, duration_sec):
    start = time.perf_counter()
    return takewhile(lambda _: time.perf_counter() - start < duration_sec, iterable)
