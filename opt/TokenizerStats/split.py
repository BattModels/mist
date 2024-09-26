from pathlib import Path
import subprocess
import shutil
import logging

logging.basicConfig(level=logging.DEBUG)


PATH = Path("/nfs/turbo/coe-venkvis/mist/realspace_v4_dev")
DST = Path("/nfs/turbo/coe-venkvis/mist/realspace_v4_dev2")

print(PATH)
for file in PATH.glob("**/*.txt"):
    rfile = file.relative_to(PATH)
    dst = DST.joinpath(rfile)

    # Copy the original file to the new location
    dst.parent.mkdir(parents=True, exist_ok=True)
    logging.debug("copy %s to %s", file, dst)
    shutil.copy(file, DST.joinpath(rfile))

    # Split the file into chunks
    prefix = dst.with_suffix("").name + "-"
    logging.debug("split %s into %sXX.txt", rfile, prefix)
    subprocess.run(
        ["split", "-n", "r/4", "-u", "--additional-suffix=.txt", str(dst), prefix],
        check=True,
        capture_output=False,
        cwd=dst.parent,
        text=True,
    )
    dst.unlink()
    logging.info("split %s", rfile)
