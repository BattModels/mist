"""Stage a matching FASMIFRA assembler and Python fragmenter."""

import argparse
import logging
import os
import platform
import shutil
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

import truststore

logger = logging.getLogger(__name__)

# Pin both tools together: current fragmenters use atom maps for cut bonds,
# which the old v1.1.0 assembler cannot interpret.
FASMIFRA_REVISION = "2f5f06024d5fbba1ae26ac374523f25341ced6c1"
OCAML_VERSION = "5.3.0"


def install_fasmifra(work_dir, output):
    work_dir, output = Path(work_dir).resolve(), Path(output).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    opam = shutil.which("opam") or str(work_dir / "opam")
    if not Path(opam).is_file():
        architecture = {"aarch64": "arm64"}.get(platform.machine(), platform.machine())
        url = (
            "https://github.com/ocaml/opam/releases/download/2.2.1/"
            f"opam-2.2.1-{architecture}-{platform.system().lower()}"
        )
        logger.info("Downloading opam from %s", url)
        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        pending = work_dir / "opam.partial"
        with (
            urllib.request.urlopen(url, context=context, timeout=60) as source,
            pending.open("wb") as target,
        ):
            shutil.copyfileobj(source, target)
        pending.chmod(0o755)
        pending.replace(opam)

    source = work_dir / f"FASMIFRA-{FASMIFRA_REVISION[:12]}"
    if not source.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--no-checkout",
                "https://github.com/UnixJunkie/FASMIFRA.git",
                str(source),
            ],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(source), "checkout", "--detach", FASMIFRA_REVISION],
        check=True,
    )
    environment = dict(
        os.environ,
        OPAMROOT=str(work_dir / "opam-root"),
        OPAMSWITCH=str(source),
        OPAMJOBS="4",
        # OPAMFETCH splits on spaces without interpreting shell quoting.
        OPAMFETCH=" ".join(
            [
                sys.executable,
                str(
                    Path(__file__).resolve().parents[3]
                    / "electrolyte_fm/utils/fetch_file.py"
                ),
                "%{url}%",
                "%{out}%",
            ]
        ),
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
    )

    def run(*command):
        subprocess.run([opam, *command], cwd=source, env=environment, check=True)

    if not (work_dir / "opam-root/config").is_file():
        run(
            "init",
            "--bare",
            "--no-setup",
            "--disable-sandboxing",
            "--yes",
            "git+https://github.com/ocaml/opam-repository.git",
        )
    if not (source / "_opam/.opam-switch/switch-config").is_file():
        run(
            "switch",
            "create",
            str(source),
            f"ocaml-base-compiler.{OCAML_VERSION}",
            "--no-install",
            "--no-depexts",
            "--yes",
        )
    # Upstream's installer skips this C++ probe; the executable does not link RDKit.
    run("install", "conf-rdkit", "--fake", "--no-depexts", "--yes")
    run("install", ".", "--deps-only", "--no-depexts", "--yes")
    run(
        "exec",
        "--",
        "dune",
        "build",
        "--profile=release",
        "src/fasmifra.exe",
        "-j",
        "4",
    )

    binary = source / "_build/default/src/fasmifra.exe"
    fragmenter = source / "bin/fasmifra_fragment.py"
    with TemporaryDirectory(dir=work_dir) as temporary:
        molecules, fragments, generated = (
            Path(temporary) / "input.smi",
            Path(temporary) / "fragments.smi",
            Path(temporary) / "output.smi",
        )
        molecules.write_text(
            "CCCCCCCCCCCCCCCCCCCC\tsmoke1\nCCCCCCCCCCCCCCCCCCCCCCCC\tsmoke2\n"
        )
        subprocess.run(
            [
                sys.executable,
                str(fragmenter),
                "-i",
                str(molecules),
                "-o",
                str(fragments),
                "-w",
                "150",
                "-n",
                "1",
                "--seed",
                "1",
            ],
            check=True,
            timeout=30,
        )
        tagged = fragments.read_text().splitlines()
        if len(tagged) != 2 or not all("[*:" in line for line in tagged):
            raise RuntimeError("FASMIFRA smoke test did not tag both input molecules")
        subprocess.run(
            [str(binary), "-i", str(fragments), "-o", str(generated), "-n", "2"],
            check=True,
            timeout=30,
        )
        assembled = generated.read_text().splitlines()
        if len(assembled) != 2 or any("*" in line for line in assembled):
            raise RuntimeError("FASMIFRA smoke test did not assemble the cut bonds")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(binary, output)
    output.chmod(0o755)
    installed_fragmenter = output.with_name("fasmifra_fragment.py")
    shutil.copyfile(fragmenter, installed_fragmenter)
    installed_fragmenter.chmod(0o755)
    logger.info("Built FASMIFRA at %s", output)
    logger.info("Installed FASMIFRA fragmenter at %s", installed_fragmenter)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    install_fasmifra(args.work_dir, args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
