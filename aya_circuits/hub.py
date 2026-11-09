"""Hugging Face Hub publishing. Every dataset, circuit, adapter, checkpoint, log and scorecard goes
under the owner's namespace (brief §5). Token comes from HF_TOKEN; nothing is pushed implicitly."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi

NAMESPACE = os.environ.get("HF_NAMESPACE", "MdSadiqMd")


def push_folder(
    local: str | Path,
    repo: str,
    repo_type: str = "dataset",
    private: bool = False,
    msg: str = "",
    ignore_patterns: list[str] | None = None,
) -> str:
    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = f"{NAMESPACE}/{repo}"
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(local),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=msg or f"update {Path(local).name}",
        ignore_patterns=ignore_patterns,
    )
    return f"https://huggingface.co/{'datasets/' if repo_type == 'dataset' else ''}{repo_id}"


def delete_paths(repo: str, repo_type: str, paths: list[str], msg: str) -> list[str]:
    """Delete the given files from the repo if they exist (one commit); returns the paths
    removed."""
    from huggingface_hub import CommitOperationDelete

    api = HfApi(token=os.environ["HF_TOKEN"])
    repo_id = f"{NAMESPACE}/{repo}"
    present = set(api.list_repo_files(repo_id, repo_type=repo_type))
    todo = [p for p in paths if p in present]
    if todo:
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=[CommitOperationDelete(path_in_repo=p) for p in todo],
            commit_message=msg,
        )
    return todo
