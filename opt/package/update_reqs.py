#!/usr/bin/env python3

from __future__ import annotations

import argparse
from io import BytesIO
from typing import Iterable

from huggingface_hub import CommitOperationAdd, HfApi, file_exists, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError


def iter_org_model_repo_ids(api: HfApi, org: str) -> Iterable[str]:
    # list_models(author=...) returns ModelInfo objects with .modelId / .id depending on version.
    for m in api.list_models(author=org):
        repo_id = getattr(m, "modelId", None) or getattr(m, "id", None)
        if repo_id:
            yield repo_id


def load_existing_requirements(repo_id: str, token: str | None) -> str | None:
    if not file_exists(repo_id, "requirements.txt", repo_type="model", token=token):
        return None
    try:
        path = hf_hub_download(
            repo_id, "requirements.txt", repo_type="model", token=token
        )
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except EntryNotFoundError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True, help="HF organization name (namespace)")
    ap.add_argument(
        "--requirements", required=True, help="Path to desired requirements.txt"
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--create-pr", action="store_true", help="Open a PR per repo (safer)"
    )
    mode.add_argument(
        "--push-direct", action="store_true", help="Commit directly to main (dangerous)"
    )
    ap.add_argument(
        "--revision", default="main", help='Base revision/branch (default: "main")'
    )
    ap.add_argument(
        "--commit-message", default="Update requirements.txt", help="Commit title"
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print actions, do not commit"
    )
    ap.add_argument(
        "--patch-append",
        action="append",
        default=[],
        help="Line(s) to append if missing (repeatable). If not provided, overwrites file exactly.",
    )
    args = ap.parse_args()

    import os

    token = os.environ.get("HF_TOKEN")

    api = HfApi(token=token)

    with open(args.requirements, "r", encoding="utf-8") as f:
        desired = f.read().strip() + "\n"

    repo_ids = list(iter_org_model_repo_ids(api, args.org))
    print(f"Found {len(repo_ids)} model repos under org={args.org}")

    for repo_id in repo_ids:
        # Decide new content
        new_content = desired
        if args.patch_append:
            existing = load_existing_requirements(repo_id, token)
            if existing is None:
                base = ""
            else:
                base = existing
            lines = [ln.rstrip("\n") for ln in base.splitlines() if ln.strip()]
            s = set(lines)
            for extra in args.patch_append:
                if extra not in s:
                    lines.append(extra)
            new_content = "\n".join(lines).strip() + "\n"

        if args.dry_run:
            print(
                f"[DRY RUN] Would update {repo_id}:requirements.txt (create_pr={args.create_pr})"
            )
            continue

        op = CommitOperationAdd(
            path_in_repo="requirements.txt",
            path_or_fileobj=BytesIO(new_content.encode("utf-8")),
        )
        api.create_commit(
            repo_id=repo_id,
            repo_type="model",
            revision=args.revision,
            operations=[op],
            commit_message=args.commit_message,
            create_pr=bool(args.create_pr),
        )
        print(f"Updated {repo_id}")

    print("Done.")


if __name__ == "__main__":
    main()
