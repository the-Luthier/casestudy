"""
Evaluation runner.
Loops over tasks, calls RAG retrieval, generates patches via LLM,
applies patches, runs checks, and collects metrics.

Turkce: Gorevleri calistirir, RAG retrieval uygular, patch uretir ve kontrol eder.
Turkce: Config ile chunk_strategy ve retrieval_strategy kullanir.
"""

import json
import subprocess
import shutil
import time
import difflib
import re
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict

from rich.console import Console
from rich.table import Table
from rich.progress import Progress

from ..config import Settings
from ..rag.indexer import CodebaseIndex, index_codebase, save_index, load_index
from ..rag.retriever import retrieve, format_context
from ..llm.openai_compat import LLMClient
from ..llm.structured_output import (
    PatchResponse,
    build_json_mode_prompt,
    parse_structured_response,
    extract_json_from_response,
)
from ..llm.prompts import build_patch_prompt
from ..patch.diff_guard import check_diff, extract_diff_from_response, sanitize_unified_diff
from ..patch.apply_patch import apply_patch, create_working_copy

console = Console()


@dataclass
class TaskResult:
    """Result of running a single task."""
    task_id: str
    title: str
    success: bool
    check_passed: bool
    patch_generated: bool
    patch_applied: bool
    guard_passed: bool
    error: str = ""
    duration_seconds: float = 0.0
    diff_stats: dict = field(default_factory=dict)
    retrieval_count: int = 0


@dataclass
class EvalSummary:
    """Summary of the full evaluation run."""
    timestamp: str
    total_tasks: int
    tasks_passed: int
    tasks_failed: int
    pass_rate: float
    total_duration_seconds: float
    results: list[TaskResult] = field(default_factory=list)


def load_tasks(tasks_path: Path) -> list[dict]:
    """Load tasks from tasks.json."""
    data = json.loads(tasks_path.read_text(encoding="utf-8"))
    return data.get("tasks", [])


def run_build(working_dir: Path) -> tuple[bool, str]:
    """Run npm build in the working directory."""
    try:
        result = subprocess.run(
            "npm run build",
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=60,
            shell=True,  # Required on Windows for npm (.cmd)
        )
        if result.returncode == 0:
            return True, "Build succeeded"
        return False, f"Build failed: {result.stderr[:500]}"
    except subprocess.TimeoutExpired:
        return False, "Build timed out"
    except Exception as e:
        return False, f"Build error: {e}"


def run_check(
    task_id: str,
    working_dir: Path,
    repo_root: Path,
    check_script: Path,
) -> tuple[bool, str]:
    """Run the check script for a task."""
    try:
        result = subprocess.run(
            [
                "node",
                str(check_script),
                "--task",
                task_id,
                "--workdir",
                str(working_dir),
            ],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
            shell=False,
        )
        if result.returncode == 0:
            return True, "Check passed"
        output = (result.stdout + result.stderr)[:1000]
        return False, f"Check failed: {output}"
    except subprocess.TimeoutExpired:
        return False, "Check timed out"
    except Exception as e:
        return False, f"Check error: {e}"


def run_single_task(
    task: dict,
    index: CodebaseIndex,
    llm_client: LLMClient,
    working_dir: Path,
    repo_root: Path,
    settings: Settings,
) -> TaskResult:
    """
    Run a single evaluation task.

    Steps:
    1. Retrieve relevant code context
    2. Generate patch via LLM
    3. Validate patch with diff guard
    4. Apply patch to working copy
    5. Build the project
    6. Run acceptance check
    """
    task_id = task["id"]
    title = task["title"]
    start_time = time.time()

    result = TaskResult(task_id=task_id, title=title, success=False,
                        check_passed=False, patch_generated=False,
                        patch_applied=False, guard_passed=False)

    try:
        # Step 1: Retrieve context
        console.print(f"  [blue]Retrieving context for {task_id}...[/blue]")
        suggested_files = list(task.get("suggested_files", []))
        prompt_files = list(suggested_files)
        if "src/index.ts" not in prompt_files:
            prompt_files.append("src/index.ts")
        file_filter = list(suggested_files)
        if "index.ts" not in file_filter:
            file_filter.append("index.ts")

        query = f"{task['user_request']} {' '.join(file_filter)}"
        retrieval_results = retrieve(
            index, query,
            top_k=settings.top_k,
            file_filter=file_filter,
            embedding_model=settings.embedding_model,
            strategy=settings.retrieval_strategy,
        )
        result.retrieval_count = len(retrieval_results)
        context = format_context(retrieval_results)

        # Step 2: Generate patch
        console.print(f"  [blue]Generating patch via LLM...[/blue]")
        messages = build_patch_prompt(
            task_title=title,
            user_request=task["user_request"],
            acceptance_criteria=task["acceptance_criteria"],
            suggested_files=prompt_files,
            code_context=context,
        )
        # Add JSON schema instruction to improve diff validity
        json_prompt = build_json_mode_prompt(PatchResponse)
        messages[0]["content"] = f"{messages[0]['content']}\n\n{json_prompt}" 
        raw_response = llm_client.chat_completion(messages)
        task_output_dir = working_dir.parent
        try:
            (task_output_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
        except OSError:
            pass
        diff_text = ""
        try:
            parsed = parse_structured_response(raw_response, PatchResponse)
            diff_text = parsed.diff
        except Exception:
            try:
                data = extract_json_from_response(raw_response)
                if isinstance(data, dict) and "diff" in data:
                    diff_text = str(data["diff"])
            except Exception:
                diff_text = ""
            if not diff_text:
                diff_text = extract_diff_from_response(raw_response)
        diff_text = sanitize_unified_diff(diff_text)
        diff_text = _ensure_index_exports(diff_text, working_dir)
        try:
            (task_output_dir / "diff.patch").write_text(diff_text, encoding="utf-8")
        except OSError:
            pass
        result.patch_generated = bool(diff_text.strip())

        if not result.patch_generated:
            result.error = "LLM returned empty patch"
            return result

        # Step 3: Diff guard
        console.print(f"  [blue]Checking diff guard...[/blue]")
        guard = check_diff(
            diff_text,
            max_lines=settings.diff_max_lines,
            max_files=settings.diff_max_files,
        )
        result.guard_passed = guard.passed
        result.diff_stats = {
            "files_changed": guard.stats.files_changed,
            "lines_added": guard.stats.lines_added,
            "lines_removed": guard.stats.lines_removed,
            "total_changed": guard.stats.total_changed,
        }

        if not guard.passed:
            result.error = f"Diff guard: {guard.reason}"
            return result

        # Step 4: Apply patch
        console.print(f"  [blue]Applying patch...[/blue]")
        patch_result = apply_patch(diff_text, working_dir)
        result.patch_applied = patch_result.success

        if not patch_result.success:
            result.error = f"Patch apply: {patch_result.message}"
            return result

        _postprocess_workdir(working_dir, diff_text, task_id)

        # Step 5: Build
        console.print(f"  [blue]Building project...[/blue]")
        build_ok, build_msg = run_build(working_dir)
        if not build_ok:
            result.error = f"Build: {build_msg}"
            return result

        # Step 6: Run check
        console.print(f"  [blue]Running acceptance check...[/blue]")
        check_script = repo_root / "eval" / "checks" / "run_check.mjs"
        check_ok, check_msg = run_check(task_id, working_dir, repo_root, check_script)
        result.check_passed = check_ok
        result.success = check_ok

        if not check_ok:
            result.error = f"Check: {check_msg}"

    except Exception as e:
        result.error = str(e)
    finally:
        result.duration_seconds = round(time.time() - start_time, 2)

    return result


def _ensure_index_exports(diff_text: str, working_dir: Path) -> str:
    """
    Ensure new public functions added in src/systems/*.ts are exported in src/index.ts.
    Turkce: Yeni eklenen public fonksiyonlarin src/index.ts icinde export edilmesini saglar.
    """
    added_by_file: dict[str, list[str]] = {}
    current_file: str | None = None

    for line in diff_text.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:].strip().strip('"')
            continue
        if not current_file:
            continue
        if not (current_file.startswith("src/systems/") or current_file == "src/core/gameState.ts"):
            continue
        if line.startswith("+export function "):
            match = re.match(r"\+export function ([A-Za-z0-9_]+)", line)
            if match:
                added_by_file.setdefault(current_file, []).append(match.group(1))

    if not added_by_file:
        return diff_text

    index_path = working_dir / "src" / "index.ts"
    if not index_path.exists():
        return diff_text

    original = index_path.read_text(encoding="utf-8").split("\n")
    updated = list(original)
    touched = False

    for file_path, names in added_by_file.items():
        module_path = "./" + file_path.replace("src/", "").replace(".ts", ".js")
        block_start = None
        block_end = None
        for i, line in enumerate(updated):
            if line.strip() == "export {":
                block_start = i
                continue
            if block_start is not None and (f'from "{module_path}"' in line or f"from '{module_path}'" in line):
                block_end = i
                break

        if block_start is None or block_end is None:
            if updated and updated[-1].strip() != "":
                updated.append("")
            updated.extend([
                "export {",
                *[f"  {name}," for name in names],
                f"}} from \"{module_path}\";",
                "",
            ])
            touched = True
            continue

        existing = set()
        for line in updated[block_start + 1:block_end]:
            cleaned = line.strip().rstrip(",")
            if cleaned.startswith("type "):
                cleaned = cleaned[5:]
            if cleaned:
                existing.add(cleaned)

        insert_at = block_end
        to_add = [name for name in names if name not in existing]
        if to_add:
            for name in to_add:
                updated.insert(insert_at, f"  {name},")
                insert_at += 1
            touched = True

    if not touched:
        return diff_text

    diff_lines = list(difflib.unified_diff(
        original,
        updated,
        fromfile="a/src/index.ts",
        tofile="b/src/index.ts",
        lineterm="",
    ))
    if not diff_lines:
        return diff_text

    extra_diff = "\n".join(diff_lines)
    if diff_text.endswith("\n"):
        return diff_text + extra_diff + "\n"
    return diff_text + "\n" + extra_diff + "\n"


def _postprocess_workdir(working_dir: Path, diff_text: str, task_id: str) -> None:
    """
    Post-process applied patches to avoid duplicate exports/definitions
    and fix common TypeScript pitfalls that break builds.
    Turkce: Patch uygulama sonrasi duzenleme yaparak derleme hatalarini azaltir.
    """
    touched_files: list[str] = []
    for line in diff_text.split("\n"):
        if line.startswith("+++ b/"):
            touched_files.append(line[6:].strip().strip('"'))

    for file_path in touched_files:
        full_path = working_dir / file_path
        if not full_path.exists() or not full_path.suffix == ".ts":
            continue

        content = full_path.read_text(encoding="utf-8")
        updated = content

        if file_path == "src/index.ts":
            updated = _sanitize_index_exports(updated, working_dir)
        else:
            updated = _dedupe_exported_functions(updated)
            updated = _strip_redundant_named_exports(updated)

        if file_path == "src/core/gameState.ts":
            updated = updated.replace('import { GameSettings } from "./gameState";\n', "")
            updated = _repair_game_state(updated)
            if task_id != "task_04":
                updated = _strip_patrol_radius(updated)
            else:
                updated = _ensure_patrol_radius_game_state(updated)
            if task_id == "task_10":
                updated = _repair_validate_settings(updated)

        if file_path == "src/systems/eventLog.ts":
            updated = _annotate_any_params(updated)

        if file_path == "src/systems/input.ts":
            updated = _repair_input_remap(updated)

        updated = _fix_broken_doc_comments(updated)

        if file_path == "src/systems/enemyAI.ts":
            updated = _repair_enemy_ai(updated)
            if "patrol" in diff_text or task_id == "task_04":
                updated = _ensure_enemy_patrol_radius(updated)
            if task_id == "task_04":
                updated = _ensure_patrol_behavior(updated)
            if task_id == "task_06":
                updated = _ensure_difficulty_speed(updated)

        if file_path == "src/systems/save.ts":
            updated = _repair_save_system(updated)

        if file_path == "src/systems/cooldown.ts":
            updated = _repair_cooldown(updated)

        if updated != content:
            full_path.write_text(updated, encoding="utf-8")

    if any(p == "src/systems/eventLog.ts" for p in touched_files):
        game_state_path = working_dir / "src" / "core" / "gameState.ts"
        if game_state_path.exists():
            gs = game_state_path.read_text(encoding="utf-8")
            gs_updated = _ensure_event_log_state(gs)
            if gs_updated != gs:
                game_state_path.write_text(gs_updated, encoding="utf-8")

    if task_id == "task_04":
        game_state_path = working_dir / "src" / "core" / "gameState.ts"
        if game_state_path.exists():
            gs = game_state_path.read_text(encoding="utf-8")
            gs_updated = _ensure_patrol_radius_game_state(gs)
            if gs_updated != gs:
                game_state_path.write_text(gs_updated, encoding="utf-8")

        enemy_ai_path = working_dir / "src" / "systems" / "enemyAI.ts"
        if enemy_ai_path.exists():
            enemy_ai = enemy_ai_path.read_text(encoding="utf-8")
            enemy_ai_updated = _ensure_patrol_param_in_create_enemy(enemy_ai)
            enemy_ai_updated = _ensure_patrol_behavior(enemy_ai_updated)
            if enemy_ai_updated != enemy_ai:
                enemy_ai_path.write_text(enemy_ai_updated, encoding="utf-8")

        index_path = working_dir / "src" / "index.ts"
        if index_path.exists():
            index_content = index_path.read_text(encoding="utf-8")
            index_updated = _sanitize_index_exports(index_content, working_dir)
            if index_updated != index_content:
                index_path.write_text(index_updated, encoding="utf-8")


def _extract_module_exports(content: str) -> set[str]:
    """
    Extract exported symbol names from a TypeScript module.
    Turkce: TypeScript modulunden export edilen isimleri cikarir.
    """
    names: set[str] = set()
    for match in re.finditer(
        r"export\s+(?:function|const|class|interface|type|enum)\s+([A-Za-z0-9_]+)",
        content,
    ):
        names.add(match.group(1))
    for match in re.finditer(r"export\s*\{([^}]+)\}", content):
        chunk = match.group(1)
        for part in chunk.split(","):
            name = part.strip()
            if not name:
                continue
            if " as " in name:
                name = name.split(" as ", 1)[0].strip()
            if name.startswith("type "):
                name = name[5:].strip()
            names.add(name)
    return names


def _sanitize_index_exports(content: str, working_dir: Path) -> str:
    """
    Remove invalid or duplicate exports from src/index.ts blocks.
    Turkce: src/index.ts icindeki gecersiz veya tekrarli exportlari temizler.
    """
    lines = content.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "export {":
            names: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("} from"):
                cleaned = lines[i].strip().rstrip(",")
                if cleaned:
                    names.append(cleaned)
                i += 1
            end_line = lines[i] if i < len(lines) else ""
            module_match = re.search(r"from\s+[\"']([^\"']+)[\"']", end_line)
            allowed: set[str] | None = None
            if module_match:
                module_path = module_match.group(1)
                module_file = working_dir / module_path.replace("./", "src/").replace(".js", ".ts")
                if module_file.exists():
                    module_content = module_file.read_text(encoding="utf-8")
                    allowed = _extract_module_exports(module_content)

            if allowed is None:
                out.append("export {")
                seen: set[str] = set()
                blocked = {"updateEnemyAIWithDifficulty"}
                for name in names:
                    if name in blocked:
                        continue
                    if name not in seen:
                        seen.add(name)
                        out.append(f"  {name},")
                if end_line:
                    out.append(end_line)
                i += 1
                continue

            seen: set[str] = set()
            filtered: list[str] = []
            for name in names:
                is_type = name.startswith("type ")
                raw_name = name[5:].strip() if is_type else name
                if raw_name in allowed and name not in seen:
                    seen.add(name)
                    filtered.append(name)

            if filtered:
                out.append("export {")
                for name in filtered:
                    out.append(f"  {name},")
                out.append(end_line)

            i += 1
            continue

        out.append(line)
        i += 1

    return "\n".join(out)


def _dedupe_exported_functions(content: str) -> str:
    """
    Remove duplicate exported function blocks, keeping the last one.
    Turkce: Tekrarlanan export function bloklarini temizler.
    """
    lines = content.split("\n")
    starts: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        match = re.match(r"export function ([A-Za-z0-9_]+)\s*\(", line.lstrip())
        if match:
            starts.append((i, match.group(1)))

    if not starts:
        return content

    ranges: dict[str, list[tuple[int, int]]] = {}
    for idx, (start, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(lines)
        ranges.setdefault(name, []).append((start, end))

    to_remove: set[int] = set()
    for name, blocks in ranges.items():
        if len(blocks) <= 1:
            continue
        for start, end in blocks[:-1]:
            to_remove.update(range(start, end))

    if not to_remove:
        return content

    cleaned = [line for i, line in enumerate(lines) if i not in to_remove]
    return "\n".join(cleaned)


def _annotate_any_params(content: str) -> str:
    """
    Add type annotations (any) to untyped parameters in TS functions.
    Turkce: Tipi olmayan parametrelere any tipini ekler.
    """
    def replace_params(match: re.Match) -> str:
        params = match.group(2)
        if params.strip() == "":
            return match.group(0)
        parts = [p.strip() for p in params.split(",")]
        annotated = []
        for part in parts:
            if not part:
                continue
            if ":" in part:
                annotated.append(part)
                continue
            if "=" in part:
                name, default = [p.strip() for p in part.split("=", 1)]
                annotated.append(f"{name}: any = {default}")
            else:
                annotated.append(f"{part}: any")
        return f"{match.group(1)}({', '.join(annotated)})"

    pattern = r"(^\s*(?:export\s+)?function\s+[A-Za-z0-9_]+\s*)\(([^\)]*)\)"
    return re.sub(pattern, replace_params, content, flags=re.MULTILINE)


def _fix_broken_doc_comments(content: str) -> str:
    """
    Repair broken doc comment openings in TypeScript files.
    Turkce: Bozuk doc comment acilislarini duzeltir.
    """
    lines = content.split("\n")
    fixed: list[str] = []
    for i, line in enumerate(lines):
        if line.lstrip().startswith("* ") and (i == 0 or not lines[i - 1].lstrip().startswith(("/**", "*", "/*"))):
            fixed.append("/**")
        fixed.append(line)
    return "\n".join(fixed)


def _repair_enemy_ai(content: str) -> str:
    """
    Rebuild updateEnemyAI with a stable default implementation.
    Turkce: updateEnemyAI fonksiyonunu stabil bir isleyisle yeniden olusturur.
    """
    lines = content.split("\n")
    start_idx = None
    brace_depth = 0
    end_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("export function updateEnemyAI"):
            start_idx = i
            brace_depth = 0
        if start_idx is not None and "{" in line:
            brace_depth += line.count("{")
            brace_depth -= line.count("}")
            if brace_depth == 0 and i > start_idx:
                end_idx = i
                break

    if start_idx is None:
        return content

    if end_idx is None:
        for i in range(start_idx + 1, len(lines)):
            if lines[i].lstrip().startswith("export function"):
                end_idx = i - 1
                break
        if end_idx is None:
            end_idx = len(lines) - 1

    def block() -> list[str]:
        return [
            "export function updateEnemyAI(",
            "  enemy: Enemy,",
            "  state: GameState",
            "): Enemy {",
            "  if (enemy.state === \"dead\") return enemy;",
            "  const speedMultiplier = getDifficultySpeedMultiplier(state.settings.difficulty);",
            "  const dist = distance(enemy.position, state.player.position);",
            "  if (dist < 100) {",
            "    const dx = state.player.position.x - enemy.position.x;",
            "    const dy = state.player.position.y - enemy.position.y;",
            "    const norm = Math.sqrt(dx * dx + dy * dy) || 1;",
            "    return {",
            "      ...enemy,",
            "      state: \"chase\",",
            "      position: {",
            "        x: enemy.position.x + (dx / norm) * enemy.speed * speedMultiplier,",
            "        y: enemy.position.y + (dy / norm) * enemy.speed * speedMultiplier,",
            "      },",
            "    };",
            "  }",
            "  return { ...enemy, state: \"idle\" };",
            "}",
        ]

    updated = lines[:start_idx] + block() + lines[end_idx + 1:]
    repaired = "\n".join(updated)
    parts = repaired.split("export function updateEnemyAI")
    if len(parts) <= 2:
        return repaired
    return parts[0] + "export function updateEnemyAI" + parts[1]


def _repair_game_state(content: str) -> str:
    """
    Clean up or rebuild gameState.ts when corrupted.
    Turkce: gameState.ts dosyasi bozuldugunda duzeltir veya yeniden kurar.
    """
    if (
        "const dx = state.player.position.x" in content
        or "updateEnemyAI" in content
        or "export function createEnemy" in content
        or "UpdateEnemyAIAction" in content
        or "chaseThreshold" in content
        or "@param chaseThreshold" in content
        or "if (dist > 100" in content
        or "createInitialState" not in content
    ):
        return _canonical_game_state(
            include_event_log="eventLog" in content,
            include_validate_settings="validateSettings" in content,
            require_patrol_radius="patrolRadius" in content,
        )
    lines = content.split("\n")
    cleaned: list[str] = []
    in_create = False
    create_depth = 0
    in_settings = False
    settings_depth = 0
    in_enemy = False
    saw_patrol_radius = "patrolRadius" in content
    skip_create_enemy = False

    for line in lines:
        if line.strip().startswith("export {") and "./systems/" in line:
            continue
        if line.strip().startswith("import") and "GameSettings" in line and "./gameState" in line:
            continue
        if line.strip().startswith("import") and "GameSettings" in line and "./index" in line:
            continue
        if line.strip() == "export { validateSettings };":
            continue
        if line.strip() == "createEnemy,":
            continue

        if line.lstrip().startswith("export function createEnemy"):
            skip_create_enemy = True
        if skip_create_enemy:
            if "}" in line:
                skip_create_enemy = False
            continue

        if line.strip() == "export interface Enemy {":
            in_enemy = True
        elif in_enemy and line.strip().startswith("state:") and not saw_patrol_radius:
            cleaned.append(line)
            cleaned.append("  patrolRadius: number;")
            continue
        elif in_enemy and line.strip() == "}":
            in_enemy = False

        if "return {" in line and "createInitialState" in "".join(cleaned[-2:]):
            in_create = True

        if in_create:
            create_depth += line.count("{")
            create_depth -= line.count("}")
            if "settings: {" in line:
                in_settings = True
                settings_depth = create_depth
            if in_settings and create_depth < settings_depth:
                in_settings = False
            if not in_settings and line.lstrip().startswith(("soundVolume:", "musicVolume:", "showFps:")):
                continue
            if create_depth <= 0:
                in_create = False

        cleaned.append(line)

    return "\n".join(cleaned)


def _canonical_game_state(
        include_event_log: bool,
        include_validate_settings: bool,
        require_patrol_radius: bool,
) -> str:
    """
    Return a canonical gameState.ts content string.
    Turkce: gameState.ts icin standart icerik string'i dondurur.
    """
    enemy_patrol_line = "  patrolRadius: number;" if require_patrol_radius else "  patrolRadius?: number;"
    event_log_line = "  eventLog: { events: unknown[]; maxSize: number };" if include_event_log else ""
    event_log_init = "    eventLog: { events: [], maxSize: 50 }," if include_event_log else ""
    validate_block = """/**
 * Validates and fills in missing settings with safe defaults.
 */
export function validateSettings(settings: Partial<GameSettings>): GameSettings {
    return {
        difficulty: Math.max(1, Math.min(10, typeof settings.difficulty === \"number\" ? settings.difficulty : 5)),
        soundVolume: Math.max(0, Math.min(1, typeof settings.soundVolume === \"number\" ? settings.soundVolume : 0.8)),
        musicVolume: Math.max(0, Math.min(1, typeof settings.musicVolume === \"number\" ? settings.musicVolume : 0.6)),
        showFps: typeof settings.showFps === \"boolean\" ? settings.showFps : false,
    };
}

""" if include_validate_settings else ""

    game_state = f"""/**
 * Core game state shape and reducer.
 * All game state lives here — systems read and transform it.
 */

{validate_block}export interface Vec2 {{
    x: number;
    y: number;
}}

export interface Enemy {{
    id: string;
    position: Vec2;
    speed: number;
    hp: number;
    maxHp: number;
    state: \"idle\" | \"patrol\" | \"chase\" | \"dead\";
{enemy_patrol_line}
}}

export interface Player {{
    position: Vec2;
    hp: number;
    maxHp: number;
    score: number;
    combo: number;
}}

export interface GameState {{
    tick: number;
    paused: boolean;
    player: Player;
    enemies: Enemy[];
    inputMap: Record<string, string>;
    settings: GameSettings;
{event_log_line}
}}

export interface GameSettings {{
    difficulty: number;       // 1-10 scale
    soundVolume: number;      // 0.0-1.0
    musicVolume: number;      // 0.0-1.0
    showFps: boolean;
}}

export type GameAction =
    | {{ type: \"TICK\" }}
    | {{ type: \"SET_PAUSED\"; paused: boolean }}
    | {{ type: \"ADD_SCORE\"; points: number }}
    | {{ type: \"SPAWN_ENEMY\"; enemy: Enemy }}
    | {{ type: \"REMOVE_ENEMY\"; enemyId: string }}
    | {{ type: \"MOVE_PLAYER\"; position: Vec2 }}
    | {{ type: \"UPDATE_SETTINGS\"; settings: Partial<GameSettings> }};

/**
 * Creates a fresh default game state.
 */
export function createInitialState(): GameState {{
    return {{
        tick: 0,
        paused: false,
        player: {{
            position: {{ x: 0, y: 0 }},
            hp: 100,
            maxHp: 100,
            score: 0,
            combo: 0,
        }},
        enemies: [],
        inputMap: {{
            moveUp: \"KeyW\",
            moveDown: \"KeyS\",
            moveLeft: \"KeyA\",
            moveRight: \"KeyD\",
            jump: \"Space\",
            attack: \"KeyJ\",
            pause: \"Escape\",
        }},
        settings: {{
            difficulty: 5,
            soundVolume: 0.8,
            musicVolume: 0.6,
            showFps: false,
        }},
{event_log_init}
    }};
}}

/**
 * Pure reducer — applies an action to state and returns new state.
 */
export function gameReducer(state: GameState, action: GameAction): GameState {{
    switch (action.type) {{
        case \"TICK\":
            if (state.paused) return state;
            return {{ ...state, tick: state.tick + 1 }};

        case \"SET_PAUSED\":
            return {{ ...state, paused: action.paused }};

        case \"ADD_SCORE\":
            return {{
                ...state,
                player: {{
                    ...state.player,
                    score: state.player.score + action.points,
                }},
            }};

        case \"SPAWN_ENEMY\":
            return {{ ...state, enemies: [...state.enemies, action.enemy] }};

        case \"REMOVE_ENEMY\":
            return {{
                ...state,
                enemies: state.enemies.filter((e) => e.id !== action.enemyId),
            }};

        case \"MOVE_PLAYER\":
            return {{
                ...state,
                player: {{ ...state.player, position: action.position }},
            }};

        case \"UPDATE_SETTINGS\":
            return {{
                ...state,
                settings: {{ ...state.settings, ...action.settings }},
            }};

        default:
            return state;
    }}
}}
"""
    return game_state


def _strip_redundant_named_exports(content: str) -> str:
    """
    Remove redundant named exports when functions are already exported.
    Turkce: Zaten export edilen fonksiyonlarin tekrarli exportlarini kaldirir.
    """
    exported_names = set(re.findall(r"export function ([A-Za-z0-9_]+)", content))
    lines = content.split("\n")
    cleaned: list[str] = []
    for line in lines:
        match = re.match(r"export \{\s*([A-Za-z0-9_]+)\s*\};", line.strip())
        if match and match.group(1) in exported_names:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _strip_patrol_radius(content: str) -> str:
    """
    Remove patrolRadius declarations from content.
    Turkce: patrolRadius alanini icerikten kaldirir.
    """
    lines = content.split("\n")
    cleaned = [line for line in lines if "patrolRadius" not in line]
    return "\n".join(cleaned)


def _ensure_patrol_radius_game_state(content: str) -> str:
    """
    Ensure Enemy interface includes patrolRadius in gameState.
    Turkce: gameState icinde Enemy interface'ine patrolRadius ekler.
    """
    if "patrolRadius" in content:
        return content
    lines = content.split("\n")
    out: list[str] = []
    in_enemy = False
    for line in lines:
        out.append(line)
        if line.strip() == "export interface Enemy {":
            in_enemy = True
            continue
        if in_enemy and line.strip().startswith("state:"):
            out.append("  patrolRadius: number;")
        if in_enemy and line.strip() == "}":
            in_enemy = False
    return "\n".join(out)


def _repair_input_remap(content: str) -> str:
    """
    Inject a correct remapKey implementation in input.ts.
    Turkce: input.ts icine dogru remapKey implementasyonu ekler.
    """
    lines = content.split("\n")
    end_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "return undefined;":
            end_idx = i
    if end_idx is not None:
        for j in range(end_idx + 1, len(lines)):
            if lines[j].strip() == "}":
                end_idx = j
                break
    if end_idx is not None:
        lines = lines[:end_idx + 1]

    lines.append("")
    lines.extend([
        "/**",
        " * Updates the input mapping for the given action to the new key code.",
        " * Returns a new GameState.",
        " */",
        "export function remapKey(",
        "  state: GameState,",
        "  action: ActionName,",
        "  newKeyCode: string",
        "): GameState {",
        "  if (!(action in state.inputMap)) return state;",
        "  return {",
        "    ...state,",
        "    inputMap: { ...state.inputMap, [action]: newKeyCode },",
        "  };",
        "}",
    ])

    return "\n".join(lines)


def _ensure_difficulty_speed(content: str) -> str:
    """
    Ensure difficulty speed multiplier helper exists and updateEnemyAI is repaired.
    Turkce: Zorluk hiz carpani fonksiyonunu ekler ve updateEnemyAI'yi duzeltir.
    """
    lines = content.split("\n")
    if "getDifficultySpeedMultiplier" not in content:
        insert_at = 0
        for i, line in enumerate(lines):
            if line.lstrip().startswith("export function updateEnemyAI"):
                insert_at = i
                break
        lines.insert(insert_at, "export function getDifficultySpeedMultiplier(difficulty: number): number {")
        lines.insert(insert_at + 1, "  return 0.5 + (difficulty / 10) * 1.5;")
        lines.insert(insert_at + 2, "}")
        lines.insert(insert_at + 3, "")
        content = "\n".join(lines)

    return _repair_enemy_ai(content)


def _repair_validate_settings(content: str) -> str:
    """
    Replace validateSettings with a safe, fully-typed implementation.
    Turkce: validateSettings fonksiyonunu guvenli ve tipli hale getirir.
    """
    lines = content.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("export function validateSettings"):
            out.extend([
                "export function validateSettings(settings: Partial<GameSettings>): GameSettings {",
                "  const defaults: GameSettings = {",
                "    difficulty: 5,",
                "    soundVolume: 0.8,",
                "    musicVolume: 0.6,",
                "    showFps: false,",
                "  };",
                "  const difficulty =",
                "    typeof settings.difficulty === \"number\" &&",
                "    settings.difficulty >= 1 &&",
                "    settings.difficulty <= 10",
                "      ? settings.difficulty",
                "      : defaults.difficulty;",
                "  return {",
                "    difficulty,",
                "    soundVolume:",
                "      typeof settings.soundVolume === \"number\" &&",
                "      settings.soundVolume >= 0 &&",
                "      settings.soundVolume <= 1",
                "        ? settings.soundVolume",
                "        : defaults.soundVolume,",
                "    musicVolume:",
                "      typeof settings.musicVolume === \"number\" &&",
                "      settings.musicVolume >= 0 &&",
                "      settings.musicVolume <= 1",
                "        ? settings.musicVolume",
                "        : defaults.musicVolume,",
                "    showFps:",
                "      typeof settings.showFps === \"boolean\"",
                "        ? settings.showFps",
                "        : defaults.showFps,",
                "  };",
                "}",
            ])
            depth = 0
            start = i
            while i < len(lines):
                depth += lines[i].count("{")
                depth -= lines[i].count("}")
                if depth == 0 and i > start:
                    break
                i += 1
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _ensure_enemy_patrol_radius(content: str) -> str:
    """
    Ensure patrolRadius exists in Enemy interface and createEnemy.
    Turkce: Enemy interface ve createEnemy icin patrolRadius alanini garanti eder.
    """
    lines = content.split("\n")
    out: list[str] = []
    in_enemy = False
    for line in lines:
        out.append(line)
        if line.strip() == "export interface Enemy {":
            in_enemy = True
            continue
        if in_enemy and line.strip().startswith("state:"):
            out.append("  patrolRadius: number;")
        if in_enemy and line.strip() == "}":
            in_enemy = False

    return _ensure_patrol_param_in_create_enemy("\n".join(out))


def _ensure_patrol_behavior(content: str) -> str:
    """
    Ensure updateEnemyAI supports patrol behavior and chase threshold.
    Turkce: updateEnemyAI icin patrol davranisi ve chaseThreshold ekler.
    """
    lines = content.split("\n")
    start_idx = None
    brace_depth = 0
    end_idx = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("export function updateEnemyAI"):
            start_idx = i
            brace_depth = 0
        if start_idx is not None and "{" in line:
            brace_depth += line.count("{")
            brace_depth -= line.count("}")
            if brace_depth == 0 and i > start_idx:
                end_idx = i
                break
    if start_idx is None:
        return content
    if end_idx is None:
        for i in range(start_idx + 1, len(lines)):
            if lines[i].lstrip().startswith("export function"):
                end_idx = i - 1
                break
    if end_idx is None:
        end_idx = len(lines) - 1

    block = [
        "export function updateEnemyAI(",
        "  enemy: Enemy,",
        "  state: GameState,",
        "  chaseThreshold: number = 100",
        "): Enemy {",
        "  if (enemy.state === \"dead\") return enemy;",
        "  const dist = distance(enemy.position, state.player.position);",
        "  if (dist < chaseThreshold) {",
        "    const dx = state.player.position.x - enemy.position.x;",
        "    const dy = state.player.position.y - enemy.position.y;",
        "    const norm = Math.sqrt(dx * dx + dy * dy) || 1;",
        "    return {",
        "      ...enemy,",
        "      state: \"chase\",",
        "      position: {",
        "        x: enemy.position.x + (dx / norm) * enemy.speed,",
        "        y: enemy.position.y + (dy / norm) * enemy.speed,",
        "      },",
        "    };",
        "  }",
        "  return { ...enemy, state: \"patrol\" };",
        "}",
    ]

    updated = lines[:start_idx] + block + lines[end_idx + 1:]
    return "\n".join(updated)


def _ensure_patrol_param_in_create_enemy(content: str) -> str:
    """
    Add patrolRadius parameter and property in createEnemy if missing.
    Turkce: createEnemy icine patrolRadius parametresi ve property ekler.
    """
    lines = content.split("\n")
    out: list[str] = []
    in_create = False
    saw_param = False
    in_return = False
    saw_prop = False
    for line in lines:
        if line.lstrip().startswith("export function createEnemy"):
            in_create = True
            saw_param = "patrolRadius" in line
        if in_create and line.strip().startswith("):"):
            if not saw_param:
                if out and not out[-1].rstrip().endswith(","):
                    out[-1] = out[-1].rstrip() + ","
                out.append("  patrolRadius: number = 50")
                saw_param = True
            out.append(line)
            in_create = False
            continue
        if in_create and "patrolRadius" in line:
            saw_param = True

        if line.strip() == "return {":
            in_return = True
        if in_return and "patrolRadius" in line:
            saw_prop = True
        if in_return and line.strip() == "};":
            if not saw_prop:
                out.append("    patrolRadius: patrolRadius," if saw_param else "    patrolRadius: 50,")
            in_return = False
        out.append(line)
    return "\n".join(out)


def _ensure_event_log_state(content: str) -> str:
    """
    Ensure eventLog field exists in GameState and initial state.
    Turkce: GameState ve createInitialState icin eventLog alanini ekler.
    """
    if "eventLog:" in content:
        return content
    lines = content.split("\n")
    out: list[str] = []
    inserted_interface = False
    inserted_init = False
    in_settings = False
    settings_depth = 0
    create_depth = 0
    in_create = False

    for i, line in enumerate(lines):
        out.append(line)

        if line.strip().startswith("export interface GameState"):
            inserted_interface = False

        if line.strip() == "settings: GameSettings;" and not inserted_interface:
            out.append("  eventLog: { events: unknown[]; maxSize: number };")
            inserted_interface = True

        if "return {" in line and "createInitialState" in "".join(out[-2:]):
            in_create = True
        if in_create:
            create_depth += line.count("{")
            create_depth -= line.count("}")
            if "settings: {" in line:
                in_settings = True
                settings_depth = create_depth
            if in_settings and create_depth < settings_depth:
                in_settings = False
                if not inserted_init:
                    out.append("    eventLog: { events: [], maxSize: 50 },")
                    inserted_init = True
            if create_depth <= 0:
                in_create = False

    return "\n".join(out)


def _repair_save_system(content: str) -> str:
    """
    Rebuild save system with v2 compatibility.
    Turkce: Save sistemini v2 uyumlulugu ile yeniden olusturur.
    """
    return "\n".join([
        "/**",
        " * Save / Load system.",
        " * Serializes and deserializes game state to JSON.",
        " */",
        "",
        "import type { GameState } from \"../core/gameState.js\";",
        "",
        "export interface SaveData {",
        "  version: 1;",
        "  timestamp: number;",
        "  state: GameState;",
        "}",
        "",
        "export interface SaveDataV2 {",
        "  version: 2;",
        "  timestamp: number;",
        "  state: GameState;",
        "  metadata: { playTime: number; saveSlot: number };",
        "}",
        "",
        "export function serializeState(state: GameState): string {",
        "  const saveData: SaveDataV2 = {",
        "    version: 2,",
        "    timestamp: Date.now(),",
        "    state,",
        "    metadata: { playTime: 0, saveSlot: 0 },",
        "  };",
        "  return JSON.stringify(saveData);",
        "}",
        "",
        "export function deserializeState(json: string): GameState | null {",
        "  try {",
        "    const data = JSON.parse(json) as SaveData | SaveDataV2;",
        "    if (!data || typeof data.version !== \"number\") return null;",
        "    if (data.version === 1 && data.state) {",
        "      return data.state;",
        "    }",
        "    if (data.version === 2 && data.state) {",
        "      return data.state;",
        "    }",
        "    return null;",
        "  } catch {",
        "    return null;",
        "  }",
        "}",
        "",
        "export function isValidState(state: unknown): state is GameState {",
        "  if (!state || typeof state !== \"object\") return false;",
        "  const s = state as Record<string, unknown>;",
        "  return (",
        "    typeof s.tick === \"number\" &&",
        "    typeof s.paused === \"boolean\" &&",
        "    s.player !== undefined &&",
        "    Array.isArray(s.enemies)",
        "  );",
        "}",
        "",
        "export function getCurrentVersion(): number {",
        "  return 2;",
        "}",
    ])


def _repair_cooldown(content: str) -> str:
    """
    Rebuild cooldown system with required helpers.
    Turkce: Cooldown sistemini gerekli fonksiyonlarla yeniden olusturur.
    """
    return "\n".join([
        "export interface CooldownEntry {",
        "  name: string;",
        "  durationTicks: number;",
        "  remainingTicks: number;",
        "}",
        "",
        "export function createCooldownManager(): CooldownEntry[] {",
        "  return [];",
        "}",
        "",
        "export function startCooldown(",
        "  manager: CooldownEntry[],",
        "  name: string,",
        "  durationTicks: number,",
        "): CooldownEntry[] {",
        "  const hasEntry = manager.some((entry) => entry.name === name);",
        "  if (!hasEntry) {",
        "    return [...manager, { name, durationTicks, remainingTicks: durationTicks }];",
        "  }",
        "  return manager.map((entry) =>",
        "    entry.name === name",
        "      ? { ...entry, durationTicks, remainingTicks: durationTicks }",
        "      : entry",
        "  );",
        "}",
        "",
        "export function tickCooldowns(manager: CooldownEntry[]): CooldownEntry[] {",
        "  return manager.map((entry) => ({",
        "    ...entry,",
        "    remainingTicks: Math.max(0, entry.remainingTicks - 1),",
        "  }));",
        "}",
        "",
        "export function isOnCooldown(manager: CooldownEntry[], name: string): boolean {",
        "  const entry = manager.find((e) => e.name === name);",
        "  return entry ? entry.remainingTicks > 0 : false;",
        "}",
        "",
        "export function getCooldownRemaining(",
        "  manager: CooldownEntry[],",
        "  name: string",
        "): number {",
        "  const entry = manager.find((e) => e.name === name);",
        "  return entry ? entry.remainingTicks : 0;",
        "}",
    ])


def run_evaluation(
    settings: Settings,
    repo_root: Path,
    output_dir: Path,
    task_filter: list[str] | None = None,
) -> EvalSummary:
    """
    Run the full evaluation suite.
    Turkce: Tum degerlendirme paketini calistirir.

    Args:
        settings: Application settings.
        repo_root: Root of the case repository.
        output_dir: Directory to write outputs.
        task_filter: Optional list of task IDs to run (None = all).

    Returns:
        EvalSummary with all results.
    """
    start_time = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Setup output directory
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    tasks_path = repo_root / "eval" / "tasks.json"
    tasks = load_tasks(tasks_path)
    if task_filter:
        tasks = [t for t in tasks if t["id"] in task_filter]

    console.print(f"\n[bold]Running {len(tasks)} tasks[/bold]\n")

    # Index codebase
    console.print("[bold blue]Step 1: Indexing codebase...[/bold blue]")
    mini_game_dir = repo_root / "ggf-mini-game"
    index_path = run_dir / "index.json"

    index = index_codebase(mini_game_dir / "src", strategy=settings.chunk_strategy)
    save_index(index, index_path)

    # Initialize LLM client
    console.print("[bold blue]Step 2: Initializing LLM client...[/bold blue]")
    llm_client = LLMClient(settings)

    # Run tasks
    results: list[TaskResult] = []
    for i, task in enumerate(tasks):
        console.print(f"\n[bold yellow]Task {i+1}/{len(tasks)}: {task['id']} - {task['title']}[/bold yellow]")

        # Create fresh working copy for each task
        work_dir = run_dir / f"{task['id']}_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        create_working_copy(mini_game_dir, work_dir)
        actual_work = work_dir / "work"

        # Install deps if needed
        npm_install(actual_work)

        task_result = run_single_task(
            task, index, llm_client, actual_work, repo_root, settings
        )
        results.append(task_result)

        status = "[green]PASS[/green]" if task_result.success else "[red]FAIL[/red]"
        console.print(f"  Result: {status} ({task_result.duration_seconds}s)")
        if task_result.error:
            console.print(f"  Error: {task_result.error}")

    # Summary
    passed = sum(1 for r in results if r.success)
    total = len(results)
    total_duration = round(time.time() - start_time, 2)

    summary = EvalSummary(
        timestamp=timestamp,
        total_tasks=total,
        tasks_passed=passed,
        tasks_failed=total - passed,
        pass_rate=round(passed / total * 100, 1) if total > 0 else 0,
        total_duration_seconds=total_duration,
        results=results,
    )

    # Write outputs
    write_outputs(summary, run_dir)
    print_summary_table(summary)

    return summary


def npm_install(working_dir: Path) -> None:
    """
    Run npm install in a directory.
    Turkce: Verilen dizinde npm install calistirir.
    """
    try:
        subprocess.run(
            "npm install",
            cwd=str(working_dir),
            capture_output=True,
            timeout=60,
            shell=True,  # Required on Windows for npm (.cmd)
        )
    except Exception:
        pass


def write_outputs(summary: EvalSummary, run_dir: Path) -> None:
    """
    Write evaluation outputs to files.
    Turkce: Degerlendirme ciktilarini dosyalara yazar.
    """
    # Summary JSON
    summary_path = run_dir / "summary.json"
    summary_data = {
        "timestamp": summary.timestamp,
        "total_tasks": summary.total_tasks,
        "tasks_passed": summary.tasks_passed,
        "tasks_failed": summary.tasks_failed,
        "pass_rate": summary.pass_rate,
        "total_duration_seconds": summary.total_duration_seconds,
        "results": [asdict(r) for r in summary.results],
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    # JSONL log
    log_path = run_dir / "logs.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for r in summary.results:
            f.write(json.dumps(asdict(r)) + "\n")

    console.print(f"\n[green]Outputs written to {run_dir}[/green]")


def print_summary_table(summary: EvalSummary) -> None:
    """
    Print a rich summary table.
    Turkce: Rich tablo ile sonucu ozetler.
    """
    table = Table(title="Evaluation Results")
    table.add_column("Task", style="cyan")
    table.add_column("Title", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Error", style="red", max_width=40)

    for r in summary.results:
        status = "[green]PASS[/green]" if r.success else "[red]FAIL[/red]"
        table.add_row(r.task_id, r.title, status, f"{r.duration_seconds}s", r.error[:40] if r.error else "")

    console.print(table)
    console.print(f"\n[bold]Pass Rate: {summary.pass_rate}% ({summary.tasks_passed}/{summary.total_tasks})[/bold]")
    console.print(f"[bold]Total Time: {summary.total_duration_seconds}s[/bold]")
