"""Keep workflow TLS verified through the operating system's certificate store."""

import importlib.util
import inspect
import io
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import truststore

from electrolyte_fm.utils.cache import cached_download

ROOT = Path(__file__).resolve().parents[1]
APPLICATIONS = [
    ("train.py", "_jsonnet"),
    ("electrolyte_fm/utils/stage_hf.py", "huggingface_hub"),
    ("opt/synth_access/main.py", "accelerate"),
    ("opt/screening/screen.py", "torch._dynamo"),
    ("opt/sterochemistry/scripts/train_tmqm_lr_cell.py", "torch"),
    ("opt/sterochemistry/scripts/test_tmqm_selected.py", "numpy"),
    ("opt/benchmarks/scripts/train_screening_head.py", "canonical_tasks"),
    ("opt/benchmarks/scripts/score_test.py", "numpy"),
    ("opt/benchmarks/scripts/materialize_moleculenet.py", "canonical_tasks"),
    ("opt/benchmarks/scripts/materialize_application.py", "numpy"),
]


def assert_verified_context(context):
    assert isinstance(context, truststore.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def scscore_downloader():
    path = ROOT / "opt/synth_access/vendor/scscore/scscore.py"
    spec = importlib.util.spec_from_file_location("workflow_scscore", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return lambda url, path: module.download_model(path, url)


@pytest.mark.parametrize("kind", ["cache", "scscore"])
def test_downloaders_verify_against_system_truststore(kind, tmp_path, monkeypatch):
    downloader = cached_download if kind == "cache" else scscore_downloader()
    destination = tmp_path / "download" / "payload"
    calls = []

    def urlopen(request, *, context):
        assert_verified_context(context)
        calls.append(request)
        return io.BytesIO(b"verified download")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    assert downloader("https://example.com/data", destination) == destination
    assert destination.read_bytes() == b"verified download"
    assert len(calls) == 1
    if kind == "cache":
        downloader("https://example.com/data", destination)
        assert len(calls) == 1


@pytest.mark.parametrize("kind", ["cache", "scscore"])
def test_certificate_failure_never_retries_without_verification(
    kind, tmp_path, monkeypatch
):
    downloader = cached_download if kind == "cache" else scscore_downloader()
    destination = tmp_path / "payload"
    calls = []

    def urlopen(request, *, context):
        assert_verified_context(context)
        calls.append(request)
        raise urllib.error.URLError(ssl.SSLCertVerificationError("untrusted"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(urllib.error.URLError, match="untrusted"):
        downloader("https://example.com/data", destination)
    assert len(calls) == 1
    assert not destination.exists()


def test_cached_download_has_no_insecure_override():
    assert "disable_ssl" not in inspect.signature(cached_download).parameters


@pytest.mark.parametrize("path,boundary", APPLICATIONS)
@pytest.mark.parametrize("run_name", ["__main__", "library_import_probe"])
def test_truststore_precedes_http_imports_only_at_app_startup(
    path, boundary, run_name, tmp_path
):
    # Stop at the first third-party import: no models, training or network IO.
    code = r"""
import builtins
import runpy
import ssl
import sys
import truststore

original_context = ssl.SSLContext
original_import = builtins.__import__

class ReachedDependency(Exception):
    pass

def intercept(name, *args, **kwargs):
    if name == sys.argv[2]:
        if sys.argv[3] == "__main__":
            assert ssl.SSLContext is truststore.SSLContext
            context = ssl.create_default_context()
            assert context.verify_mode == ssl.CERT_REQUIRED
            assert context.check_hostname
        else:
            assert ssl.SSLContext is original_context
        raise ReachedDependency
    return original_import(name, *args, **kwargs)

builtins.__import__ = intercept
try:
    runpy.run_path(sys.argv[1], run_name=sys.argv[3])
except ReachedDependency:
    pass
else:
    raise AssertionError("did not reach the expected dependency")
"""
    subprocess.run(
        [sys.executable, "-c", code, str(ROOT / path), boundary, run_name],
        cwd=tmp_path,
        check=True,
        timeout=30,
    )


def test_injection_reaches_requests_and_aiohttp_default_verified_contexts():
    code = """
import ssl
import truststore
truststore.inject_into_ssl()
import requests
import aiohttp.connector
from urllib3.util.ssl_ import create_urllib3_context

for context in (
    ssl.create_default_context(),
    create_urllib3_context(),
    aiohttp.connector._SSL_CONTEXT_VERIFIED,
):
    assert isinstance(context, truststore.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_qmist_bootstrap_uses_native_system_certificates():
    source = (ROOT / "opt/qmist/install.sh").read_text()
    assert source.index("export UV_SYSTEM_CERTS=true") < source.index("uv sync")
    assert "--insecure" not in source
    subprocess.run(["bash", "-n", str(ROOT / "opt/qmist/install.sh")], check=True)
