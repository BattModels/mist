import json
import logging
import multiprocessing
from types import SimpleNamespace

import pytest

from src.utils import configure_logging, merge_rank_logs


def write_rank(logfile, rank, world_size):
    logger = logging.getLogger("screening-log-test")
    fabric = SimpleNamespace(global_rank=rank, world_size=world_size)
    configure_logging(fabric, logfile, logger=logger)
    for index in range(100):
        logger.info({"index": index, "payload": "x" * 8192})
    logging.shutdown()


def test_parallel_ranks_publish_complete_jsonl(tmp_path):
    logfile = tmp_path / "screen.jsonl"
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=write_rank, args=(logfile, rank, 8)) for rank in range(8)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=30)
        assert worker.exitcode == 0
    merge_rank_logs(logfile, 8)
    records = [json.loads(line) for line in logfile.read_text().splitlines()]
    assert len(records) == 800
    for rank in range(8):
        assert [row["index"] for row in records if row["global_rank"] == rank] == list(range(100))


def test_missing_rank_does_not_replace_published_trace(tmp_path):
    logfile = tmp_path / "screen.jsonl"
    logfile.write_text("previous trace\n")
    write_rank(logfile, 0, 2)
    with pytest.raises(FileNotFoundError):
        merge_rank_logs(logfile, 2)
    assert logfile.read_text() == "previous trace\n"
