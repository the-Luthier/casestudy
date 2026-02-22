"""
Patch application using git apply.
Applies unified diffs to a working copy of the codebase.
"""

import subprocess
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
import re

from rich.console import Console

console = Console()


@dataclass
class PatchResult:
    """Result of applying a patch."""
    success: bool
    message: str
    patch_file: str = ""


def apply_patch(
    diff_text: str,
    working_dir: Path,
    dry_run: bool = False,
) -> PatchResult:
    """
    Apply a unified diff to a working directory using git apply.
    Turkce: Birlesik diff'i git apply kullanarak calisma dizinine uygular.

    Args:
        diff_text: The unified diff text.
        working_dir: Directory to apply the patch in.
        dry_run: If True, only check if the patch applies cleanly.

    Returns:
        PatchResult with success status and message.
    """
    if not diff_text.strip():
        return PatchResult(success=False, message="Empty patch")

    diff_text = _normalize_diff_for_workdir(diff_text, working_dir)

    # Write diff to temp file
    try:
        patch_file = working_dir / ".tmp_patch.diff"
        patch_file.write_text(diff_text, encoding="utf-8")
    except OSError as e:
        return PatchResult(success=False, message=f"Failed to write patch file: {e}")

    try:
        # Check if git is available
        cmd = ["git", "apply"]
        if dry_run:
            cmd.append("--check")
        cmd.append(str(patch_file))

        result = subprocess.run(
            cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            action = "would apply" if dry_run else "applied"
            return PatchResult(
                success=True,
                message=f"Patch {action} successfully",
                patch_file=str(patch_file),
            )
        else:
            error = result.stderr.strip() or result.stdout.strip()
            manual = _manual_apply(diff_text, working_dir, dry_run)
            if manual.success:
                return PatchResult(
                    success=True,
                    message="Patch applied manually after git apply failure",
                    patch_file=str(patch_file),
                )
            return PatchResult(
                success=False,
                message=f"git apply failed: {error}",
                patch_file=str(patch_file),
            )

    except FileNotFoundError:
        # git not available, try manual apply
        return _manual_apply(diff_text, working_dir, dry_run)
    except subprocess.TimeoutExpired:
        return PatchResult(success=False, message="Patch application timed out")
    except Exception as e:
        return PatchResult(success=False, message=f"Unexpected error: {e}")
    finally:
        # Clean up temp file
        try:
            if "patch_file" in locals():
                patch_file.unlink(missing_ok=True)
        except OSError:
            pass


def _manual_apply(
    diff_text: str,
    working_dir: Path,
    dry_run: bool = False,
) -> PatchResult:
    """
    Fallback manual patch application for environments without git.
    Handles simple unified diffs only.
    Turkce: Git olmayan ortamlarda basit diff'leri manuel uygular.
    """
    import re

    current_file = None
    hunks: dict[str, list[tuple[int, list[str], list[str], list[str]]]] = {}

    lines = diff_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # New file
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            if current_file not in hunks:
                hunks[current_file] = []
            i += 1
            continue

        # Skip --- lines
        if line.startswith("--- a/"):
            i += 1
            continue

        # Hunk header
        if line.startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if match and current_file:
                old_start = int(match.group(1))
                removes: list[str] = []
                adds: list[str] = []
                adds_only: list[str] = []
                i += 1

                while i < len(lines):
                    hline = lines[i]
                    if hline.startswith("@@") or hline.startswith("diff ") or hline.startswith("--- ") or hline.startswith("+++ "):
                        break
                    if hline.startswith("-"):
                        removes.append(hline[1:])
                    elif hline.startswith("+"):
                        adds.append(hline[1:])
                        adds_only.append(hline[1:])
                    elif hline.startswith(" "):
                        removes.append(hline[1:])
                        adds.append(hline[1:])
                    i += 1

                hunks[current_file].append((old_start, removes, adds, adds_only))
                continue

        i += 1

    if not hunks:
        return PatchResult(success=False, message="No valid hunks found in patch")

    if dry_run:
        return PatchResult(success=True, message="Dry run: patch appears valid")

    # Apply hunks
    try:
        for file_path, file_hunks in hunks.items():
            full_path = working_dir / file_path

            if full_path.exists():
                content = full_path.read_text(encoding="utf-8")
                file_lines = content.split("\n")
            else:
                # New file
                full_path.parent.mkdir(parents=True, exist_ok=True)
                file_lines = []

            # Apply hunks in reverse order to preserve line numbers
            for old_start, removes, adds, adds_only in reversed(file_hunks):
                idx = _find_hunk_index(file_lines, removes, old_start)
                if idx is None:
                    if adds_only and _adds_already_present(file_lines, adds_only):
                        continue
                    insert_at = _find_fallback_insert_index(file_lines, removes, old_start)
                    for j, add_line in enumerate(adds_only or adds):
                        file_lines.insert(insert_at + j, add_line)
                    continue
                del file_lines[idx:idx + len(removes)]
                for j, add_line in enumerate(adds):
                    file_lines.insert(idx + j, add_line)

            full_path.write_text("\n".join(file_lines), encoding="utf-8")

        return PatchResult(success=True, message="Patch applied manually")
    except Exception as e:
        return PatchResult(success=False, message=f"Manual apply failed: {e}")


def _find_hunk_index(file_lines: list[str], removes: list[str], old_start: int) -> int | None:
    """
    Find the best match position for a hunk removal block.
    Turkce: Hunk silme blogu icin en uygun eslesme konumunu bulur.
    """
    if not removes:
        idx = max(min(old_start - 1, len(file_lines)), 0)
        return idx

    preferred = max(old_start - 1, 0)
    if file_lines[preferred:preferred + len(removes)] == removes:
        return preferred

    matches: list[int] = []
    max_start = len(file_lines) - len(removes)
    if max_start >= 0:
        for i in range(max_start + 1):
            if file_lines[i:i + len(removes)] == removes:
                matches.append(i)
    if not matches:
        return None

    return min(matches, key=lambda i: abs(i - preferred))


def _adds_already_present(file_lines: list[str], adds_only: list[str]) -> bool:
    """
    Check if all add-only lines already exist in the target file.
    Turkce: Sadece ekleme satirlarinin hedef dosyada mevcut olup olmadigini kontrol eder.
    """
    if not adds_only:
        return True
    for add_line in adds_only:
        if add_line not in file_lines:
            return False
    return True


def _find_fallback_insert_index(file_lines: list[str], removes: list[str], old_start: int) -> int:
    """
    Find a fallback insertion index when a hunk cannot be matched.
    Turkce: Hunk eslesmezse alternatif ekleme indeksini bulur.
    """
    preferred = max(min(old_start - 1, len(file_lines)), 0)
    for line in reversed(removes):
        if line.strip() == "":
            continue
        for i in range(len(file_lines) - 1, -1, -1):
            if file_lines[i] == line:
                return i + 1
    return preferred


def _normalize_diff_for_workdir(diff_text: str, working_dir: Path) -> str:
    """
    Try to re-anchor hunk headers using file context in the working directory.
    Turkce: Hunk basliklarini calisma dizinindeki baglama gore yeniden hizalar.
    """
    lines = diff_text.split("\n")
    out: list[str] = []
    current_file: str | None = None
    file_cache: dict[str, list[str]] = {}
    i = 0

    while i < len(lines):
        line = lines[i]
        if line.startswith("--- a/"):
            out.append(line)
            i += 1
            continue
        if line.startswith("+++ b/"):
            current_file = line[6:].strip().strip('"')
            out.append(line)
            i += 1
            continue
        if line.startswith("@@"):
            header = line
            hunk_lines: list[str] = []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                if i == len(lines) - 1 and next_line == "":
                    break
                if next_line.startswith("@@") or next_line.startswith("diff --git") or next_line.startswith("--- ") or next_line.startswith("+++ "):
                    break
                hunk_lines.append(next_line)
                i += 1

            header = _reanchor_hunk_header(header, hunk_lines, current_file, working_dir, file_cache)
            out.append(header)
            out.extend(hunk_lines)
            continue

        out.append(line)
        i += 1

    normalized = "\n".join(out)
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def _reanchor_hunk_header(
    header: str,
    hunk_lines: list[str],
    current_file: str | None,
    working_dir: Path,
    file_cache: dict[str, list[str]],
) -> str:
    """
    Recompute hunk header line numbers using nearby context lines.
    Turkce: Yakindaki baglam satirlarini kullanarak hunk basligini yeniden hesaplar.
    """
    match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", header)
    if not match:
        return header

    old_start = int(match.group(1))
    new_start = int(match.group(3))

    old_count = 0
    new_count = 0
    removed_count = 0
    added_count = 0
    context_lines: list[str] = []
    for hline in hunk_lines:
        if hline.startswith("-"):
            old_count += 1
            removed_count += 1
        elif hline.startswith("+"):
            new_count += 1
            added_count += 1
        elif hline.startswith(" "):
            old_count += 1
            new_count += 1
            context_lines.append(hline[1:])
        else:
            old_count += 1
            new_count += 1
            context_lines.append(hline)

    if current_file and context_lines:
        if current_file not in file_cache:
            file_path = working_dir / current_file
            if file_path.exists():
                file_cache[current_file] = file_path.read_text(encoding="utf-8").split("\n")
            else:
                file_cache[current_file] = []

        file_lines = file_cache.get(current_file, [])
        matches: list[int] = []
        max_start = len(file_lines) - len(context_lines)
        if max_start >= 0:
            for idx in range(max_start + 1):
                if file_lines[idx:idx + len(context_lines)] == context_lines:
                    matches.append(idx)
        if matches:
            if removed_count == 0 and added_count > 0 and len(context_lines) <= 2 and len(matches) > 1:
                old_start = matches[-1] + 1
            else:
                closest = min(matches, key=lambda idx: abs(idx - (old_start - 1)))
                old_start = closest + 1
            new_start = old_start

    return f"@@ -{old_start},{old_count} +{new_start},{new_count} @@"


def create_working_copy(source_dir: Path, target_dir: Path) -> Path:
    """
    Create a working copy of the source directory.
    Turkce: Kaynak dizinin calisma kopyasini olusturur.

    Args:
        source_dir: Source directory to copy.
        target_dir: Target parent directory.

    Returns:
        Path to the working copy.
    """
    work_dir = target_dir / "work"
    if work_dir.exists():
        shutil.rmtree(work_dir)

    shutil.copytree(
        source_dir, work_dir, dirs_exist_ok=False,
        ignore=shutil.ignore_patterns(".git", "__pycache__"),
    )
    console.print(f"[blue]Created working copy at {work_dir}[/blue]")
    return work_dir
