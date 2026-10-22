import importlib.util
import io
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_fasmifra.py"
spec = importlib.util.spec_from_file_location("install_fasmifra", SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.fixture
def build(tmp_path, monkeypatch):
    work, output = tmp_path / "work", tmp_path / "vendor/fasmifra"
    source = work / f"FASMIFRA-{installer.FASMIFRA_REVISION[:12]}"
    opam = tmp_path / "opam"
    opam.write_bytes(b"opam")
    monkeypatch.setattr(installer.shutil, "which", lambda name: str(opam))
    calls = []

    def run(command, **kwargs):
        assert kwargs["check"] is True
        calls.append(command)
        if "env" in kwargs:
            assert kwargs["env"]["OPAMFETCH"].split(" ") == [
                sys.executable,
                str(SCRIPT.parents[3] / "electrolyte_fm/utils/fetch_file.py"),
                "%{url}%",
                "%{out}%",
            ]
        if command[0] == "git":
            if command[1] == "clone":
                assert "--no-checkout" in command
                fragmenter = Path(command[-1]) / "bin/fasmifra_fragment.py"
                fragmenter.parent.mkdir(parents=True)
                fragmenter.write_text("#!/usr/bin/env python3\n# upstream fragmenter\n")
            else:
                assert command == [
                    "git",
                    "-C",
                    str(source),
                    "checkout",
                    "--detach",
                    installer.FASMIFRA_REVISION,
                ]
        elif "dune" in command:
            assert kwargs["env"]["OPAMROOT"] == str(work / "opam-root")
            binary = source / "_build/default/src/fasmifra.exe"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"native executable")
        elif "-o" in command:
            if command[0] == sys.executable:
                assert command[1] == str(source / "bin/fasmifra_fragment.py")
                result = "CCCC[*:1][*:1]CCCC\tsmoke1\nCCCC[*:1][*:1]CCCCC\tsmoke2\n"
            else:
                result = "CCCCCCCC\tgenmol_1\nCCCCCCCCC\tgenmol_2\n"
            Path(command[command.index("-o") + 1]).write_text(result)

    monkeypatch.setattr(installer.subprocess, "run", run)
    return work, output, calls


def test_clones_builds_and_installs_executable(build):
    work, output, calls = build
    installer.install_fasmifra(work, output)
    assert output.read_bytes() == b"native executable"
    assert output.stat().st_mode & 0o777 == 0o755
    assert any("--deps-only" in call for call in calls)
    assert any("conf-rdkit" in call and "--fake" in call for call in calls)
    assert any(
        f"ocaml-base-compiler.{installer.OCAML_VERSION}" in call for call in calls
    )
    assert any(
        "init" in call and "git+https://github.com/ocaml/opam-repository.git" in call
        for call in calls
    )
    fragmenter = output.with_name("fasmifra_fragment.py")
    assert fragmenter.read_text() == "#!/usr/bin/env python3\n# upstream fragmenter\n"
    assert fragmenter.stat().st_mode & 0o777 == 0o755
    assert {path.name for path in output.parent.iterdir()} == {
        "fasmifra",
        "fasmifra_fragment.py",
    }


@pytest.mark.parametrize("stage", ["fragmentation", "assembly"])
def test_smoke_test_rejects_incompatible_tools(build, monkeypatch, stage):
    work, output, _ = build
    output.parent.mkdir()
    output.write_bytes(b"previous binary")
    fragmenter = output.with_name("fasmifra_fragment.py")
    fragmenter.write_text("previous fragmenter")
    run = installer.subprocess.run

    def incompatible(command, **kwargs):
        run(command, **kwargs)
        if "-o" not in command:
            return
        if stage == "fragmentation" and command[0] == sys.executable:
            result = "CCCCCCCC\tsmoke1\nCCCCCCCC\tsmoke2\n"
        elif stage == "assembly" and command[0] != sys.executable:
            result = "CCCC[*:1][*:1]CCCC\tgenmol_1\nCCCC[*:1][*:1]CCCC\tgenmol_2\n"
        else:
            return
        Path(command[command.index("-o") + 1]).write_text(result)

    monkeypatch.setattr(installer.subprocess, "run", incompatible)
    with pytest.raises(RuntimeError, match="FASMIFRA smoke test"):
        installer.install_fasmifra(work, output)
    assert output.read_bytes() == b"previous binary"
    assert fragmenter.read_text() == "previous fragmenter"


def test_opam_always_uses_truststore_downloader(build, monkeypatch):
    work, output, _ = build
    monkeypatch.setenv("OPAMFETCH", "curl")
    installer.install_fasmifra(work, output)


def test_bootstraps_missing_opam_with_truststore(build, monkeypatch):
    work, output, _ = build
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    monkeypatch.setattr(installer.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")

    def download(url, *, context, timeout):
        assert url.endswith("opam-2.2.1-arm64-linux")
        assert isinstance(context, installer.truststore.SSLContext)
        return io.BytesIO(b"downloaded opam")

    monkeypatch.setattr(installer.urllib.request, "urlopen", download)
    installer.install_fasmifra(work, output)
    assert (work / "opam").read_bytes() == b"downloaded opam"
    assert (work / "opam").stat().st_mode & 0o777 == 0o755


def test_build_failure_does_not_replace_existing_binary(build, monkeypatch):
    work, output, _ = build
    output.parent.mkdir()
    output.write_bytes(b"previous binary")
    run = installer.subprocess.run

    def fail(command, **kwargs):
        if "dune" in command:
            raise subprocess.CalledProcessError(1, command)
        return run(command, **kwargs)

    monkeypatch.setattr(installer.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        installer.install_fasmifra(work, output)
    assert output.read_bytes() == b"previous binary"
