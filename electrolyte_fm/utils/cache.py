from pathlib import Path


def find_member(tar, name):
    for m in tar.getmembers():
        # Strip the first path component
        p = Path(*str(m.name).split("/")[1:])
        if str(p) == name:
            return m

    return None


def extract_file(archive, name, output: Path, **kwargs):
    import tarfile
    from shutil import move
    from tempfile import TemporaryDirectory

    with tarfile.open(archive) as fid:
        member = find_member(fid, name)
        assert member is not None
        with TemporaryDirectory() as tmp:
            fid.extract(member, filter="data", path=tmp, **kwargs)
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            move(str(Path(tmp).joinpath(member.name)), str(output))


def cached_github_archive(repo, commit, file):
    cache = Path(__file__).parent.parent.parent.joinpath(".cache", "smiles-tokenizers")
    cache.mkdir(exist_ok=True, parents=True)

    archive = Path(
        "smiles-tokenizers", repo.replace("/", "-"), commit, "archive.tar.gz"
    )
    url = f"https://github.com/{repo}/archive/{commit}.tar.gz"
    archive = cached_download(url, archive)

    cached_path = archive.parent.joinpath(file)
    if not cached_path.is_file():
        extract_file(archive, file, cached_path)

    return cached_path


def cached_download(url: str, path: Path) -> Path:
    path = Path(path)
    cache = Path(__file__).parent.parent.parent.joinpath(".cache")
    cached_file = cache.joinpath(path)
    cached_file.parent.mkdir(exist_ok=True, parents=True)
    if not cached_file.exists():
        import urllib

        with urllib.request.urlopen(url) as fid:
            cached_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cached_file, "wb") as out:
                out.write(fid.read())

    return cached_file
