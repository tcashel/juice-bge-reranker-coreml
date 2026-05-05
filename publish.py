"""
Push converted .mlpackage variants to a Hugging Face Hub model repo.

For each requested variant, stages a directory with the layout the Swift consumer
expects (a single `model.mlpackage`, tokenizer files at the root so
`swift-transformers` finds `tokenizer.json` immediately, the MODEL_CARD.md as
the repo's README.md, and the provenance sidecar), uploads it, then creates a
Hugging Face git tag `v{X}-{variant}` on the resulting commit.

After both variants land, also creates an annotated **local** git tag `v{X}` on
the source repo's HEAD so the published artifact traces back to a specific git
commit. The tag is created locally only — push it explicitly with
`git push origin v{X}` once you're satisfied. Refuses if the working tree is
dirty or the tag already exists; pass `--no-git-tag` to skip this step.

Refuses to do any HF push without `--confirm`. The plan explicitly requires
explicit human acknowledgement before publishing.

Usage:
    HUGGINGFACE_TOKEN=hf_xxx pixi run python publish.py --variant both --tag v0.1 --confirm
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError

DEFAULT_REPO_ID = "tcashel/bge-reranker-base-coreml"
DEFAULT_BUILD_DIR = Path("build")
ROOT = Path(__file__).resolve().parent
MODEL_CARD = ROOT / "MODEL_CARD.md"
EVAL_RESULTS_DIR = ROOT / ".eval_results"

VARIANTS = ("ane", "cpugpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    p.add_argument("--tag", required=True, help="Release tag base, e.g. 'v0.1'. The variant suffix is appended.")
    p.add_argument(
        "--variant",
        choices=("ane", "cpugpu", "both"),
        default="both",
        help="Which variant(s) to publish.",
    )
    p.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    p.add_argument("--dry-run", action="store_true", help="Stage the upload directory but do not push.")
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Required for any non-dry-run push. Without this, the script refuses.",
    )
    p.add_argument(
        "--no-git-tag",
        action="store_true",
        help="Skip creating the local git tag after HF publish. Useful for partial republishes.",
    )
    return p.parse_args()


def variant_artifact(build_dir: Path, variant: str) -> Path:
    return build_dir / f"bge-reranker-base-{variant}.mlpackage"


def variant_provenance(build_dir: Path, variant: str) -> Path:
    return build_dir / f"{variant}_provenance.json"


def stage_variant(build_dir: Path, variant: str, dest: Path) -> None:
    """Lay out <variant>-specific files for upload as a single HF commit.

    Layout (everything at the repo root so swift-transformers' AutoTokenizer.from(modelFolder:)
    finds tokenizer.json without a subpath):
      README.md                  <- MODEL_CARD.md
      model.mlpackage/...        <- the variant's .mlpackage, renamed for stable consumer paths
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json
      sentencepiece.bpe.model
      provenance.json            <- {variant}_provenance.json
      .eval_results/*.yaml       <- per-task quality eval results (HF model-index format)
    """
    artifact = variant_artifact(build_dir, variant)
    if not artifact.exists():
        raise FileNotFoundError(f"Missing artifact: {artifact}")
    provenance = variant_provenance(build_dir, variant)
    if not provenance.exists():
        raise FileNotFoundError(f"Missing provenance: {provenance}. Re-run convert.py.")
    tokenizer_dir = build_dir / "tokenizer"
    if not tokenizer_dir.exists():
        raise FileNotFoundError(f"Missing tokenizer dir: {tokenizer_dir}. Re-run convert.py.")
    if not MODEL_CARD.exists():
        raise FileNotFoundError(f"Missing MODEL_CARD.md at {MODEL_CARD}.")

    dest.mkdir(parents=True, exist_ok=True)
    # Rename .mlpackage to a stable name so the Swift side can hardcode "model.mlpackage".
    shutil.copytree(artifact, dest / "model.mlpackage")
    # Tokenizer files at root.
    for f in tokenizer_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    # Provenance + readme.
    shutil.copy2(provenance, dest / "provenance.json")
    shutil.copy2(MODEL_CARD, dest / "README.md")
    # Quality eval results (HF reads .eval_results/*.yaml at the repo root). Same numbers
    # apply to both variants — FP16 weights are bit-identical between -ane and -cpugpu.
    if EVAL_RESULTS_DIR.is_dir():
        eval_yamls = sorted(EVAL_RESULTS_DIR.glob("*.yaml"))
        if eval_yamls:
            (dest / ".eval_results").mkdir(exist_ok=True)
            for f in eval_yamls:
                shutil.copy2(f, dest / ".eval_results" / f.name)


def _git(args: list[str]) -> str:
    """Run a git command in the source repo and return stdout."""
    result = subprocess.run(["git", *args], capture_output=True, text=True, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout


def check_git_state(tag: str) -> str:
    """Verify the source repo is in a publishable state. Returns the HEAD SHA being tagged.

    Refuses to proceed if:
      - The working tree is dirty (published artifacts must trace to a clean commit).
      - The local git tag already exists (would silently re-point on push).

    Warns (but doesn't refuse) if HEAD is not on main — best practice is to publish
    from a merged commit, but a maintainer may have a reason to publish from a branch.
    """
    dirty = _git(["status", "--porcelain"]).strip()
    if dirty:
        raise SystemExit(
            "publish.py refuses: working tree is dirty. Commit or stash before publishing —\n"
            "the published artifact must be traceable back to a clean git commit.\n\n"
            f"Dirty paths:\n{dirty}\n\n"
            "(Pass --no-git-tag to skip the git-tag step entirely if you really need to.)"
        )
    if _git(["tag", "--list", tag]).strip():
        raise SystemExit(
            f"publish.py refuses: local git tag {tag!r} already exists.\n"
            f"  - Delete it first if you're re-publishing: git tag -d {tag}\n"
            f"  - Or pass --no-git-tag to skip the git-tag step (HF tags are independent)."
        )
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    if branch != "main":
        print(f"[warning] HEAD is on branch {branch!r}, not 'main'. Best practice is to publish from main after merging the PR.")
    head = _git(["rev-parse", "HEAD"]).strip()
    return head


def create_git_tag(tag: str, head: str, hf_tags: list[str], repo_id: str) -> None:
    """Create an annotated local git tag on HEAD pointing at the published artifact set."""
    msg_lines = [
        f"Publish bge-reranker-base-coreml {tag}",
        "",
        f"Hugging Face: https://huggingface.co/{repo_id}",
        "Tags pushed:",
    ]
    msg_lines.extend(f"  - {t}" for t in hf_tags)
    msg = "\n".join(msg_lines)
    _git(["tag", "-a", tag, "-m", msg])
    print(f"\nCreated local git tag {tag} on {head[:12]}.")
    print(f"To push it to GitHub: git push origin {tag}")


def ensure_repo(api: HfApi, repo_id: str) -> None:
    try:
        api.repo_info(repo_id, repo_type="model")
    except RepositoryNotFoundError as e:
        raise SystemExit(
            f"Hugging Face repo not found: {repo_id}\nCreate it at https://huggingface.co/new (Model, public, MIT) before publishing."
        ) from e


def push_variant(
    api: HfApi,
    repo_id: str,
    variant: str,
    tag_base: str,
    staging_dir: Path,
    *,
    dry_run: bool,
) -> str:
    tag = f"{tag_base}-{variant}"
    commit_msg = f"publish {variant} variant @ {tag}"
    if dry_run:
        print(f"[dry-run] would upload {staging_dir} -> {repo_id} and tag {tag}")
        return tag
    print(f"  uploading staging dir -> {repo_id} (this is the commit that will be tagged)")
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_msg,
    )
    print(f"  creating tag {tag}")
    api.create_tag(
        repo_id=repo_id,
        tag=tag,
        repo_type="model",
        tag_message=commit_msg,
        exist_ok=False,
    )
    return tag


def main() -> int:
    args = parse_args()

    if not args.dry_run and not args.confirm:
        print(
            "publish.py refuses to push without --confirm. Re-run with --dry-run first; "
            "review the staged layout; then re-run with --confirm to proceed.",
            file=sys.stderr,
        )
        return 2

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not args.dry_run and not token:
        print(
            "HUGGINGFACE_TOKEN (or HF_TOKEN) not set. Run `huggingface-cli login` or export the token.",
            file=sys.stderr,
        )
        return 2

    targets = VARIANTS if args.variant == "both" else (args.variant,)
    print(f"repo: {args.repo_id}")
    print(f"variants: {list(targets)}   tag base: {args.tag}")
    print(f"build dir: {args.build_dir.resolve()}")

    # Gate on git state before any HF work so a half-publish can't ship without a
    # corresponding source-repo tag. Skip when --no-git-tag.
    head_sha: str | None = None
    if not args.no_git_tag:
        if args.dry_run:
            head_sha = _git(["rev-parse", "HEAD"]).strip()
            print(f"[dry-run] would tag git HEAD ({head_sha[:12]}) as {args.tag} after HF push")
        else:
            head_sha = check_git_state(args.tag)
            print(f"git HEAD: {head_sha[:12]}  (will tag as {args.tag} after HF push succeeds)")

    api = HfApi(token=token) if token else HfApi()
    if not args.dry_run:
        ensure_repo(api, args.repo_id)

    pushed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="bge-coreml-publish-") as tmp:
        tmp_root = Path(tmp)
        for variant in targets:
            print(f"\n[{variant}] staging...")
            staging = tmp_root / variant
            stage_variant(args.build_dir, variant, staging)
            tag = push_variant(api, args.repo_id, variant, args.tag, staging, dry_run=args.dry_run)
            pushed.append(tag)

    print()
    if args.dry_run:
        print("Dry-run complete. Re-run with --confirm to publish.")
    else:
        print("Published tags:")
        for t in pushed:
            print(f"  https://huggingface.co/{args.repo_id}/tree/{t}")
        if not args.no_git_tag and head_sha is not None:
            create_git_tag(args.tag, head_sha, pushed, args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
