import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from settings_manager import load_settings, save_settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hub folders of master simulations across environment change-rate values."
    )
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--script", type=str, default="simulation_entry.py")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--start-rate", type=float, default=None)
    parser.add_argument("--end-rate", type=float, default=None)
    parser.add_argument("--step", type=float, default=None)
    parser.add_argument("--species-threshold", type=int, default=None)
    parser.add_argument("--max-masters", type=int, default=None)
    parser.add_argument(
        "--hub-select",
        action="store_true",
        help="Force hub selector UI (new hub or continue existing).",
    )
    parser.add_argument(
        "--continue-hub",
        type=int,
        default=None,
        help="Continue a specific existing hub index (e.g. --continue-hub 3).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip matplotlib plot generation and only write CSV/JSON outputs.",
    )
    parser.add_argument(
        "--no-screen",
        action="store_true",
        help="Disable the live hub dashboard window.",
    )
    parser.add_argument(
        "--screen-hold-seconds",
        type=float,
        default=-1.0,
        help=(
            "Keep the dashboard open after completion "
            "(-1 = until manually closed, 0 = do not hold, >0 = hold for N seconds)."
        ),
    )
    return parser.parse_args()


def _hub_defaults_from_settings() -> dict:
    settings = load_settings()
    hub_cfg = settings.get("hub", {}) if isinstance(settings, dict) else {}
    if not isinstance(hub_cfg, dict):
        hub_cfg = {}
    return {
        "start_rate": float(hub_cfg.get("start_rate", 0.5)),
        "end_rate": float(hub_cfg.get("end_rate", 1.5)),
        "step": float(hub_cfg.get("step", 0.01)),
        "species_threshold": int(hub_cfg.get("species_threshold", 100000)),
        "max_masters": int(hub_cfg.get("max_masters", 101)),
    }


def _copy_to_clipboard(text: str) -> bool:
    payload = str(text) if text is not None else ""
    if payload == "":
        return False
    try:
        import pyperclip  # pylint: disable=import-outside-toplevel

        pyperclip.copy(payload)
        return True
    except Exception:
        pass

    commands: list[list[str]] = []
    if sys.platform == "darwin":
        commands.append(["pbcopy"])
    if os.name == "nt":
        commands.append(["clip"])
    commands.extend(
        [
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ]
    )
    for cmd in commands:
        try:
            subprocess.run(
                cmd,
                input=payload.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except Exception:
            continue
    return False


def _open_path_with_default_app(path: Path) -> bool:
    if not isinstance(path, Path):
        return False
    try:
        if not path.exists():
            return False
    except Exception:
        return False
    target = str(path)
    commands: list[list[str]] = []
    if sys.platform == "darwin":
        commands.append(["open", target])
    elif os.name == "nt":
        commands.append(["cmd", "/c", "start", "", target])
    else:
        commands.append(["xdg-open", target])
        commands.append(["gio", "open", target])
    for cmd in commands:
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            continue
    return False


def _rate_values(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        return []
    if end < start:
        start, end = end, start
    out = []
    cur = start
    guard = 0
    while cur <= end + (step * 0.5):
        out.append(round(cur, 6))
        cur += step
        guard += 1
        if guard > 100000:
            break
    return out


def _hub_container_dir(results_root: Path) -> Path:
    return Path(results_root) / "hub"


def _master_container_dir(results_root: Path) -> Path:
    return Path(results_root) / "master"


def _normal_sim_container_dir(results_root: Path) -> Path:
    return Path(results_root) / "normal simulatinos"


def _hub_search_roots(results_root: Path) -> list[Path]:
    roots = []
    seen = set()
    for candidate in (_hub_container_dir(results_root), Path(results_root)):
        try:
            key = str(candidate.resolve())
        except Exception:
            key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        roots.append(candidate)
    return roots


def _allocate_hub_index(results_root: Path) -> int:
    settings = load_settings()
    try:
        current = int(settings.get("num_tries_hub", 0))
    except Exception:
        current = 0
    current = max(0, current)
    hub_target_root = _hub_container_dir(results_root)
    hub_target_root.mkdir(parents=True, exist_ok=True)
    default_root = _hub_container_dir(Path("results"))
    try:
        same_root = default_root.resolve() == hub_target_root.resolve()
    except Exception:
        same_root = str(default_root) == str(hub_target_root)

    candidate = current
    while True:
        in_target = hub_target_root / f"hub_{candidate}"
        in_default = default_root / f"hub_{candidate}"
        if in_target.exists():
            candidate += 1
            continue
        if (not same_root) and in_default.exists():
            candidate += 1
            continue
        break

    settings["num_tries_hub"] = candidate + 1
    save_settings(settings)
    return candidate


def _rate_label(rate: float) -> str:
    return f"env_{rate:.2f}".replace(".", "p")


def _parse_hub_id(path: Path):
    if not path.name.startswith("hub_"):
        return None
    try:
        return int(path.name.split("_", 1)[1])
    except Exception:
        return None


def _parse_master_id(path: Path):
    if not path.name.startswith("master_"):
        return None
    try:
        return int(path.name.split("_", 1)[1])
    except Exception:
        return None


def _collect_existing_master_ids(results_root: Path) -> set[int]:
    ids: set[int] = set()
    if not results_root.exists():
        return ids
    for path in results_root.rglob("master_*"):
        if not path.is_dir():
            continue
        run_id = _parse_master_id(path)
        if run_id is None:
            continue
        ids.add(int(run_id))
    return ids


def _collect_hub_runs(results_root: Path) -> list[dict]:
    hubs = []
    results_root = Path(results_root)
    if not results_root.exists():
        return hubs
    seen_hub_idxs = set()
    for search_root in _hub_search_roots(results_root):
        if not search_root.is_dir():
            continue
        for path in sorted(search_root.glob("hub_*")):
            if not path.is_dir():
                continue
            hub_idx = _parse_hub_id(path)
            if hub_idx is None:
                continue
            if int(hub_idx) in seen_hub_idxs:
                continue
            seen_hub_idxs.add(int(hub_idx))
            meta_path = path / "hub_meta.json"
            meta = {}
            if meta_path.exists():
                try:
                    loaded = json.loads(meta_path.read_text())
                    if isinstance(loaded, dict):
                        meta = loaded
                except Exception:
                    meta = {}
            rates = meta.get("rates")
            if not isinstance(rates, list):
                rates = []
            total_steps = len(rates)
            steps = meta.get("steps")
            if not isinstance(steps, list):
                steps = []
            ok_indices = set()
            for step in steps:
                if not isinstance(step, dict):
                    continue
                if str(step.get("status", "")) != "ok":
                    continue
                try:
                    idx = int(step.get("step_index"))
                except Exception:
                    continue
                if idx < 0:
                    continue
                ok_indices.add(idx)
            completed_steps = len(ok_indices)
            if total_steps > 0:
                completed_steps = min(completed_steps, total_steps)
            hubs.append(
                {
                    "hub_idx": int(hub_idx),
                    "hub_dir": path,
                    "status": str(meta.get("status", "unknown")),
                    "total_steps": int(total_steps),
                    "completed_steps": int(completed_steps),
                    "created_at": meta.get("created_at"),
                    "meta_path": meta_path,
                }
            )
    hubs.sort(key=lambda item: int(item["hub_idx"]))
    return hubs


def _ensure_csv_with_header(path: Path, header_line: str) -> None:
    if path.exists():
        try:
            if path.stat().st_size > 0:
                return
        except Exception:
            pass
    try:
        path.write_text(header_line)
    except Exception:
        pass


def _write_json_file(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
        return
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
    path.write_text(text, encoding="utf-8")


def _hub_point_triples_from_rows(rows: list[dict]) -> list[tuple[float, float, float]]:
    flattened: list[tuple[float, float, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        env_rate = row.get("env_rate")
        points = row.get("points")
        if (not _is_number(env_rate)) or (not isinstance(points, list)):
            continue
        env = float(env_rate)
        for point in points:
            if not isinstance(point, (tuple, list)) or len(point) < 2:
                continue
            evo_val = point[0]
            fit_val = point[1]
            if not (_is_number(evo_val) and _is_number(fit_val)):
                continue
            flattened.append((env, float(evo_val), float(fit_val)))
    flattened.sort(key=lambda item: (item[0], item[1], item[2]))
    return flattened


def _write_hub_all_points_csv(path: Path, rows: list[dict]) -> None:
    flattened = _hub_point_triples_from_rows(rows)
    try:
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["evo rate", "enviormentChangeRate", "fitness"])
            for env, evo, fitness in flattened:
                writer.writerow([evo, env, fitness])
    except Exception:
        pass


def _write_hub_fit_equations_csv(path: Path, rows: list[dict]) -> None:
    try:
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "enviorment change rate",
                    "master run",
                    "apex x",
                    "apex y",
                    "sigma left",
                    "sigma right",
                    "r2",
                    "equation",
                ]
            )
            ordered_rows = sorted(
                [row for row in rows if isinstance(row, dict)],
                key=lambda row: (
                    int(row.get("step_index"))
                    if _is_number(row.get("step_index"))
                    else 10**9
                ),
            )
            for row in ordered_rows:
                fit = row.get("fit")
                env_rate = row.get("env_rate")
                if (not isinstance(fit, dict)) or (not _is_number(env_rate)):
                    continue
                master_run_num = row.get("master_run_num")
                writer.writerow(
                    [
                        float(env_rate),
                        (int(master_run_num) if _is_number(master_run_num) else ""),
                        (fit.get("apex_x") if _is_number(fit.get("apex_x")) else ""),
                        (fit.get("apex_y") if _is_number(fit.get("apex_y")) else ""),
                        (fit.get("sigma_left") if _is_number(fit.get("sigma_left")) else ""),
                        (fit.get("sigma_right") if _is_number(fit.get("sigma_right")) else ""),
                        (fit.get("r2") if _is_number(fit.get("r2")) else ""),
                        str(fit.get("equation", "")),
                    ]
                )
    except Exception:
        pass


def _sync_hub_meta_steps_from_rows(hub_meta: dict, rows: list[dict]) -> bool:
    if not isinstance(hub_meta, dict):
        return False
    steps = hub_meta.get("steps")
    if not isinstance(steps, list):
        return False

    step_by_idx: dict[int, dict] = {}
    for step in steps:
        if not isinstance(step, dict):
            continue
        raw_idx = step.get("step_index")
        if not _is_number(raw_idx):
            continue
        step_by_idx[int(raw_idx)] = step

    changed = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_idx = row.get("step_index")
        if not _is_number(raw_idx):
            continue
        step = step_by_idx.get(int(raw_idx))
        if not isinstance(step, dict):
            continue

        updates = {
            "fit": (row.get("fit") if isinstance(row.get("fit"), dict) else None),
            "master_run_num": (
                int(row.get("master_run_num"))
                if _is_number(row.get("master_run_num"))
                else None
            ),
            "master_dir": (
                str(row.get("master_dir"))
                if isinstance(row.get("master_dir"), str) and str(row.get("master_dir")).strip()
                else None
            ),
            "run_nums": (
                [int(v) for v in row.get("run_nums", []) if _is_number(v)]
                if isinstance(row.get("run_nums"), list)
                else []
            ),
            "max_species": (
                float(row.get("max_species")) if _is_number(row.get("max_species")) else None
            ),
            "total_species": (
                float(row.get("total_species")) if _is_number(row.get("total_species")) else None
            ),
            "max_frames": (
                float(row.get("max_frames")) if _is_number(row.get("max_frames")) else None
            ),
            "apex_evolution_rate": (
                float(row.get("apex_evolution_rate"))
                if _is_number(row.get("apex_evolution_rate"))
                else None
            ),
            "apex_fitness": (
                float(row.get("apex_fitness")) if _is_number(row.get("apex_fitness")) else None
            ),
            "duration_s": (
                float(row.get("duration_s")) if _is_number(row.get("duration_s")) else None
            ),
            "point_count": (
                int(row.get("point_count"))
                if _is_number(row.get("point_count"))
                else (
                    len(row.get("points"))
                    if isinstance(row.get("points"), list)
                    else 0
                )
            ),
        }
        for key, value in updates.items():
            if step.get(key) != value:
                step[key] = value
                changed = True

    if changed:
        hub_meta["updated_at"] = time.time()
    return changed


def _fitness_weight_count(fitness: float) -> int:
    try:
        value = float(fitness)
    except Exception:
        return 0
    if value <= 0:
        return 0
    return max(1, int(round(value)))


_MAX_WEIGHTED_POINT_COPIES = 200_000


def _hub_all_points_weighted_by_fitness(rows: list[dict]) -> list[tuple[float, float, float]]:
    flattened = _hub_point_triples_from_rows(rows)
    weighted_specs: list[tuple[float, float, float, int]] = []
    total_copies = 0
    for env, evo, fitness in flattened:
        copies = _fitness_weight_count(fitness)
        if copies <= 0:
            continue
        weighted_specs.append((env, evo, fitness, int(copies)))
        total_copies += int(copies)
    if total_copies <= 0:
        return []

    if total_copies <= _MAX_WEIGHTED_POINT_COPIES:
        weighted: list[tuple[float, float, float]] = []
        for env, evo, fitness, copies in weighted_specs:
            weighted.extend([(env, evo, fitness)] * copies)
        return weighted

    # Keep the weighted view bounded by scaling copy counts proportionally.
    scale = float(_MAX_WEIGHTED_POINT_COPIES) / float(total_copies)
    weighted = []
    fractions: list[tuple[float, int]] = []
    for idx, (env, evo, fitness, copies) in enumerate(weighted_specs):
        scaled = float(copies) * scale
        base = int(math.floor(scaled))
        if base > 0:
            weighted.extend([(env, evo, fitness)] * base)
        fractions.append((scaled - float(base), idx))

    remaining = max(0, _MAX_WEIGHTED_POINT_COPIES - len(weighted))
    if remaining > 0 and fractions:
        fractions.sort(key=lambda item: item[0], reverse=True)
        for _, idx in fractions[:remaining]:
            env, evo, fitness, _ = weighted_specs[idx]
            weighted.append((env, evo, fitness))
    return weighted


def _select_hub_run_ui(results_root: Path):
    hub_rows = _collect_hub_runs(results_root)
    try:
        import pygame  # pylint: disable=import-outside-toplevel
    except Exception:
        return {"mode": "new"}

    pygame.init()
    try:
        screen_w = 840
        screen_h = 560
        screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("Hub Selector")
        font = pygame.font.SysFont("Consolas", 22)
        small = pygame.font.SysFont("Consolas", 18)
        tiny = pygame.font.SysFont("Consolas", 15)
        clock = pygame.time.Clock()

        rows = [{"mode": "new", "label": "Create New Hub"}]
        for row in hub_rows:
            done = int(row.get("completed_steps", 0))
            total = int(row.get("total_steps", 0))
            status = str(row.get("status", "unknown"))
            rows.append(
                {
                    "mode": "continue",
                    "hub_idx": int(row["hub_idx"]),
                    "hub_dir": row["hub_dir"],
                    "created_at": row.get("created_at"),
                    "status": status,
                    "completed_steps": done,
                    "total_steps": total,
                    "label": f"Continue hub_{int(row['hub_idx'])} ({done}/{total}) [{status}]",
                }
            )
        preview_cache: dict[str, list[tuple[float, float, float]]] = {}

        def _preview_points_for_hub(hub_dir: Path) -> list[tuple[float, float, float]]:
            key = str(hub_dir)
            cached = preview_cache.get(key)
            if cached is not None:
                return cached

            points: list[tuple[float, float, float]] = []
            all_points_path = hub_dir / "hub_all_points.csv"
            if all_points_path.exists():
                try:
                    with open(all_points_path, newline="") as handle:
                        reader = csv.DictReader(handle)
                        for row in reader:
                            if not isinstance(row, dict):
                                continue
                            env_val = row.get("enviormentChangeRate")
                            evo_val = row.get("evo rate")
                            fit_val = row.get("fitness")
                            if (not _is_number(env_val)) or (not _is_number(evo_val)):
                                continue
                            fitness = float(fit_val) if _is_number(fit_val) else 1.0
                            points.append((float(env_val), float(evo_val), float(fitness)))
                except Exception:
                    points = []

            if not points:
                summary_path = hub_dir / "hub_summary.csv"
                if summary_path.exists():
                    try:
                        with open(summary_path, newline="") as handle:
                            reader = csv.DictReader(handle)
                            for row in reader:
                                if not isinstance(row, dict):
                                    continue
                                env_val = row.get("enviorment change rate")
                                evo_val = row.get("apex evolution rate")
                                fit_val = row.get("fitness")
                                if (not _is_number(env_val)) or (not _is_number(evo_val)):
                                    continue
                                fitness = float(fit_val) if _is_number(fit_val) else 1.0
                                points.append((float(env_val), float(evo_val), float(fitness)))
                    except Exception:
                        points = []

            points.sort(key=lambda item: (item[0], item[1], item[2]))
            preview_cache[key] = points
            return points

        selected = 0
        scroll = 0
        list_top = 82
        list_bottom = screen_h - 86
        line_h = small.get_height() + 8
        visible = max(1, (list_bottom - list_top) // line_h)
        left_x = 20
        left_w = 430
        detail_x = left_x + left_w + 20
        detail_w = screen_w - detail_x - 20
        choose_rect = pygame.Rect(detail_x, screen_h - 72, detail_w, 32)
        cancel_rect = pygame.Rect(screen_w - 94, 14, 74, 28)

        def _ensure_visible() -> None:
            nonlocal scroll
            if selected < scroll:
                scroll = selected
            elif selected >= scroll + visible:
                scroll = selected - visible + 1
            scroll = max(0, min(scroll, max(0, len(rows) - visible)))

        _ensure_visible()
        while True:
            selected = max(0, min(selected, len(rows) - 1))
            current = rows[selected]

            def _selected_row():
                idx = max(0, min(int(selected), len(rows) - 1))
                return rows[idx]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key == pygame.K_UP:
                        selected = (selected - 1) % len(rows)
                        _ensure_visible()
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(rows)
                        _ensure_visible()
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        return _selected_row()
                if event.type == pygame.MOUSEWHEEL:
                    if event.y > 0:
                        selected = (selected - 1) % len(rows)
                    elif event.y < 0:
                        selected = (selected + 1) % len(rows)
                    _ensure_visible()
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    mx, my = event.pos
                    if cancel_rect.collidepoint(mx, my):
                        return None
                    if choose_rect.collidepoint(mx, my):
                        return _selected_row()
                    if left_x <= mx <= left_x + left_w and list_top <= my <= list_bottom:
                        idx = (my - list_top) // line_h + scroll
                        if 0 <= idx < len(rows):
                            selected = int(idx)
                            _ensure_visible()

            screen.fill((18, 18, 18))
            title = font.render("Hub Selector", True, (230, 230, 230))
            screen.blit(title, (20, 20))

            pygame.draw.rect(screen, (60, 60, 60), cancel_rect)
            pygame.draw.rect(screen, (160, 160, 160), cancel_rect, 1)
            cancel_text = tiny.render("Cancel", True, (230, 230, 230))
            screen.blit(
                cancel_text,
                (
                    cancel_rect.x + (cancel_rect.width - cancel_text.get_width()) // 2,
                    cancel_rect.y + 6,
                ),
            )

            pygame.draw.rect(screen, (26, 28, 32), (left_x, list_top, left_w, list_bottom - list_top))
            pygame.draw.rect(screen, (70, 74, 86), (left_x, list_top, left_w, list_bottom - list_top), 1)
            for idx in range(scroll, min(len(rows), scroll + visible)):
                row = rows[idx]
                y = list_top + (idx - scroll) * line_h
                if idx == selected:
                    pygame.draw.rect(screen, (40, 44, 54), (left_x + 2, y - 2, left_w - 4, line_h))
                color = (0, 215, 255) if idx == selected else (220, 220, 220)
                txt = small.render(str(row.get("label", "")), True, color)
                screen.blit(txt, (left_x + 8, y))

            pygame.draw.rect(screen, (26, 28, 32), (detail_x, list_top, detail_w, list_bottom - list_top))
            pygame.draw.rect(screen, (70, 74, 86), (detail_x, list_top, detail_w, list_bottom - list_top), 1)

            detail_lines = []
            if current.get("mode") == "new":
                detail_lines = [
                    "Mode: New hub run",
                    "A new hub index will be allocated.",
                    f"Root: {results_root}",
                ]
            else:
                created_at = current.get("created_at")
                created_txt = "-"
                try:
                    created_txt = datetime.fromtimestamp(float(created_at)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass
                detail_lines = [
                    f"Mode: Continue hub_{int(current.get('hub_idx'))}",
                    f"Status: {current.get('status', 'unknown')}",
                    f"Progress: {current.get('completed_steps', 0)}/{current.get('total_steps', 0)}",
                    f"Created: {created_txt}",
                    f"Path: {current.get('hub_dir')}",
                ]
            yy = list_top + 10
            for line in detail_lines:
                line_txt = tiny.render(str(line), True, (205, 205, 205))
                screen.blit(line_txt, (detail_x + 8, yy))
                yy += line_h

            preview_title = tiny.render("Hub graph preview", True, (188, 205, 228))
            screen.blit(preview_title, (detail_x + 8, yy + 4))
            preview_rect = pygame.Rect(
                detail_x + 8,
                yy + 22,
                detail_w - 16,
                max(90, list_bottom - (yy + 22) - 8),
            )
            pygame.draw.rect(screen, (16, 18, 24), preview_rect)
            pygame.draw.rect(screen, (66, 70, 82), preview_rect, 1)

            if current.get("mode") == "continue":
                hub_dir = current.get("hub_dir")
                points = _preview_points_for_hub(hub_dir) if isinstance(hub_dir, Path) else []
                if points:
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    fits = [p[2] for p in points]
                    raw_min_x = min(xs)
                    raw_max_x = max(xs)
                    raw_min_y = min(ys)
                    raw_max_y = max(ys)
                    min_x = raw_min_x
                    max_x = raw_max_x
                    min_y = raw_min_y
                    max_y = raw_max_y
                    if raw_max_x <= raw_min_x:
                        min_x = raw_min_x - 0.05
                        max_x = raw_max_x + 0.05
                    else:
                        x_pad = (raw_max_x - raw_min_x) * 0.04
                        min_x -= x_pad
                        max_x += x_pad
                    if raw_max_y <= raw_min_y:
                        min_y = raw_min_y - 0.05
                        max_y = raw_max_y + 0.05
                    else:
                        y_pad = (raw_max_y - raw_min_y) * 0.04
                        min_y -= y_pad
                        max_y += y_pad
                    fit_min = min(fits) if fits else 0.0
                    fit_max = max(fits) if fits else 1.0
                    fit_denom = max(1e-9, fit_max - fit_min)

                    plot = pygame.Rect(
                        preview_rect.x + 34,
                        preview_rect.y + 8,
                        preview_rect.width - 42,
                        preview_rect.height - 16,
                    )
                    pygame.draw.rect(screen, (12, 14, 19), plot)
                    pygame.draw.rect(screen, (56, 60, 70), plot, 1)

                    def _to_px(xv: float, yv: float) -> tuple[int, int]:
                        px = plot.x + int(((xv - min_x) / (max_x - min_x)) * plot.width)
                        py = plot.y + plot.height - int(((yv - min_y) / (max_y - min_y)) * plot.height)
                        return px, py

                    model = _linear_fit(xs, ys)
                    if model is not None:
                        sample = []
                        for i in range(100):
                            xv = min_x + ((max_x - min_x) * float(i) / 99.0)
                            yv = (float(model[0]) * xv) + float(model[1])
                            sample.append((xv, yv))
                        for i in range(1, len(sample)):
                            pygame.draw.line(
                                screen,
                                (242, 188, 96),
                                _to_px(sample[i - 1][0], sample[i - 1][1]),
                                _to_px(sample[i][0], sample[i][1]),
                                2,
                            )

                    for env_val, evo_val, fitness in points:
                        n = max(0.0, min(1.0, (float(fitness) - fit_min) / fit_denom))
                        radius = 1 + int(round(3 * n))
                        color = (
                            40 + int(210 * n),
                            128 + int(95 * n),
                            236 - int(166 * n),
                        )
                        pygame.draw.circle(screen, color, _to_px(float(env_val), float(evo_val)), radius)

                    max_y_txt = tiny.render(f"{raw_max_y:.3f}", True, (145, 145, 145))
                    min_y_txt = tiny.render(f"{raw_min_y:.3f}", True, (145, 145, 145))
                    min_x_txt = tiny.render(f"{raw_min_x:.3f}", True, (145, 145, 145))
                    max_x_txt = tiny.render(f"{raw_max_x:.3f}", True, (145, 145, 145))
                    screen.blit(max_y_txt, (plot.x - 32, plot.y - 2))
                    screen.blit(min_y_txt, (plot.x - 32, plot.bottom - 12))
                    screen.blit(min_x_txt, (plot.x, plot.bottom + 2))
                    screen.blit(max_x_txt, (plot.right - max_x_txt.get_width(), plot.bottom + 2))
                else:
                    no_data_txt = tiny.render("No hub points available yet.", True, (165, 165, 165))
                    screen.blit(no_data_txt, (preview_rect.x + 8, preview_rect.y + 8))
            else:
                new_mode_txt = tiny.render("Preview appears when a hub is selected.", True, (165, 165, 165))
                screen.blit(new_mode_txt, (preview_rect.x + 8, preview_rect.y + 8))

            btn_label = "Choose New Hub" if current.get("mode") == "new" else "Continue Selected Hub"
            pygame.draw.rect(screen, (42, 42, 46), choose_rect)
            pygame.draw.rect(screen, (170, 170, 170), choose_rect, 1)
            choose_text = small.render(btn_label, True, (230, 230, 230))
            screen.blit(
                choose_text,
                (
                    choose_rect.x + (choose_rect.width - choose_text.get_width()) // 2,
                    choose_rect.y + 6,
                ),
            )

            hint = tiny.render("Up/Down select  Enter choose  Esc cancel", True, (170, 170, 170))
            screen.blit(hint, (20, screen_h - 46))
            pygame.display.flip()
            clock.tick(30)
    except Exception:
        return {"mode": "new"}
    finally:
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass


def _plan_master_ids(start_id: int, count: int, existing_ids: set[int]) -> list[int]:
    out = []
    candidate = max(0, int(start_id))
    used = set(int(x) for x in existing_ids)
    while len(out) < count:
        if candidate not in used:
            out.append(candidate)
            used.add(candidate)
        candidate += 1
        if candidate > 10_000_000:
            break
    return out


def _format_id_span(ids: list[int]) -> str:
    if not ids:
        return "none"
    ordered = sorted(set(int(x) for x in ids))
    if len(ordered) == 1:
        return str(ordered[0])
    contiguous = all((ordered[i] + 1) == ordered[i + 1] for i in range(len(ordered) - 1))
    if contiguous:
        return f"{ordered[0]}-{ordered[-1]}"
    return ",".join(str(x) for x in ordered)


def _reserve_global_counters(
    settings_snapshot: dict,
    reserved_master_count: int,
    reserved_sim_count: int,
    planned_master_ids: list[int],
) -> tuple[int, int]:
    if not isinstance(settings_snapshot, dict):
        settings_snapshot = {}
    try:
        current_master = int(settings_snapshot.get("num_tries_master", 0))
    except Exception:
        current_master = 0
    try:
        current_sim = int(settings_snapshot.get("num_tries", 0))
    except Exception:
        current_sim = 0
    current_master = max(0, current_master)
    current_sim = max(0, current_sim)
    reserved_master_count = max(0, int(reserved_master_count))
    reserved_sim_count = max(0, int(reserved_sim_count))

    planned_max_next = None
    try:
        numeric_planned = [int(v) for v in planned_master_ids if isinstance(v, int)]
        if numeric_planned:
            planned_max_next = max(numeric_planned) + 1
    except Exception:
        planned_max_next = None

    target_master = current_master + reserved_master_count
    if isinstance(planned_max_next, int):
        target_master = max(target_master, planned_max_next)
    target_sim = current_sim + reserved_sim_count

    settings_snapshot["num_tries_master"] = int(target_master)
    settings_snapshot["num_tries"] = int(target_sim)
    save_settings(settings_snapshot)
    return int(target_master), int(target_sim)


def _master_dir_by_run_num(results_dir: Path, run_num: int | None) -> Path | None:
    if run_num is None:
        return None
    try:
        run_id = int(run_num)
    except Exception:
        return None
    candidate = results_dir / f"master_{run_id}"
    if candidate.is_dir():
        return candidate
    return None


def _select_master_dir(results_dir: Path, preferred_run_nums: list[int | None]) -> Path | None:
    seen: set[int] = set()
    for raw_run_num in preferred_run_nums:
        if raw_run_num is None:
            continue
        try:
            run_num = int(raw_run_num)
        except Exception:
            continue
        if run_num in seen:
            continue
        seen.add(run_num)
        direct = _master_dir_by_run_num(results_dir, run_num)
        if direct is not None:
            return direct
    return _latest_master_dir(results_dir)


def _latest_master_dir(results_dir: Path) -> Path | None:
    best = None
    best_num = -1
    for path in results_dir.glob("master_*"):
        if not path.is_dir():
            continue
        try:
            run_num = int(path.name.split("_", 1)[1])
        except Exception:
            continue
        if run_num > best_num:
            best_num = run_num
            best = path
    return best


def _master_run_nums(master_dir: Path) -> list[int]:
    meta_path = master_dir / "master_meta.json"
    if not meta_path.exists():
        return []
    try:
        payload = json.loads(meta_path.read_text())
    except Exception:
        return []
    out = []
    raw_runs = payload.get("run_nums", []) if isinstance(payload, dict) else []
    if not isinstance(raw_runs, list):
        return out
    for val in raw_runs:
        try:
            out.append(int(val))
        except Exception:
            continue
    return sorted(set(out))


def _read_species_from_run_meta(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    species = payload.get("amnt_of_species")
    if not isinstance(species, (int, float)):
        return None
    species_f = float(species)
    if not math.isfinite(species_f) or species_f < 0:
        return None
    return species_f


def _read_elapsed_from_run_meta(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    elapsed = payload.get("elapsed_seconds")
    if not isinstance(elapsed, (int, float)):
        return None
    elapsed_f = float(elapsed)
    if not math.isfinite(elapsed_f) or elapsed_f < 0:
        return None
    return elapsed_f


def _read_frame_from_run_meta(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    frame_count = payload.get("frame_count")
    if not isinstance(frame_count, (int, float)):
        return None
    frame_f = float(frame_count)
    if not math.isfinite(frame_f) or frame_f < 0:
        return None
    return frame_f


def _max_species(results_dir: Path, run_nums: list[int]) -> float | None:
    best = None
    for run_num in run_nums:
        species = _read_species_from_run_meta(results_dir / str(run_num) / "run_meta.json")
        if species is None:
            continue
        best = species if best is None else max(best, species)
    return best


def _total_species(results_dir: Path, run_nums: list[int]) -> float | None:
    total = 0.0
    saw_any = False
    for run_num in run_nums:
        species = _read_species_from_run_meta(results_dir / str(run_num) / "run_meta.json")
        if species is None:
            continue
        total += float(species)
        saw_any = True
    if not saw_any:
        return None
    return float(total)


def _max_elapsed_seconds(results_dir: Path, run_nums: list[int]) -> float | None:
    best = None
    for run_num in run_nums:
        elapsed = _read_elapsed_from_run_meta(results_dir / str(run_num) / "run_meta.json")
        if elapsed is None:
            continue
        best = elapsed if best is None else max(best, elapsed)
    return best


def _max_frames(results_dir: Path, run_nums: list[int]) -> float | None:
    best = None
    for run_num in run_nums:
        frames = _read_frame_from_run_meta(results_dir / str(run_num) / "run_meta.json")
        if frames is None:
            continue
        best = frames if best is None else max(best, frames)
    return best


def _extract_points_from_csv(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    points = []
    try:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                try:
                    x = float(row.get("evolution rate", ""))
                    y = float(row.get("arithmetic mean length lived", ""))
                except Exception:
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    points.append((x, y))
    except Exception:
        return []
    return points


def _master_points(master_dir: Path, run_nums: list[int]) -> list[tuple[float, float]]:
    # Prefer the master-level parsed arithmetic file for graph points.
    # This is the source of truth for: evolution rate + arithmetic mean length lived.
    preferred_master_paths: list[Path] = []
    name_variants = ("Simulatino", "Simulationo", "Simulation", "Simulatin")
    for suffix in ("_Log.csv", "_log.csv"):
        for variant in name_variants:
            preferred_master_paths.append(
                master_dir / f"parsedArithmeticMean{variant}{master_dir.name}{suffix}"
            )
    for master_path in preferred_master_paths:
        points = _extract_points_from_csv(master_path)
        if points:
            return points

    # Fallback: any non-step parsed arithmetic master file with this master suffix.
    for candidate in sorted(master_dir.glob(f"parsedArithmeticMean*{master_dir.name}_*.csv")):
        name = candidate.name.lower()
        if "_step" in name:
            continue
        points = _extract_points_from_csv(candidate)
        if points:
            return points

    # Final fallback for older runs that only have combined outputs.
    combined_path = master_dir / f"combinedArithmeticMeanSimulatino{master_dir.name}_Log.csv"
    points = _extract_points_from_csv(combined_path)
    if points:
        return points

    parent = master_dir.parent
    merged = []
    for run_num in run_nums:
        run_path = parent / str(run_num) / f"parsedArithmeticMeanSimulatino{run_num}_Log.csv"
        merged.extend(_extract_points_from_csv(run_path))
    return merged


def _combine_master_logs_for_update(
    results_dir: Path,
    master_dir: Path,
    master_label: str,
    run_nums: list[int],
) -> None:
    master_raw = master_dir / "raw_data"
    master_raw.mkdir(parents=True, exist_ok=True)
    out_path = master_raw / f"simulation_log_{master_label}.csv"
    wrote_header = False
    with open(out_path, "w", newline="") as out_handle:
        writer = csv.writer(out_handle)
        for run_num in run_nums:
            raw_dir = results_dir / str(run_num) / "raw_data"
            base = raw_dir / f"simulation_log_{run_num}.csv"
            part_files = sorted(raw_dir.glob(f"simulation_log_{run_num}_part*.csv"))
            for path in [base] + part_files:
                if not path.exists():
                    continue
                with open(path, newline="") as in_handle:
                    reader = csv.reader(in_handle)
                    try:
                        header = next(reader)
                    except StopIteration:
                        continue
                    if not wrote_header:
                        writer.writerow(header)
                        wrote_header = True
                    for row in reader:
                        writer.writerow(row)


def _combine_master_mean_kind_for_update(
    results_dir: Path,
    master_dir: Path,
    master_label: str,
    run_nums: list[int],
    kind: str,
    fieldnames: list[str],
) -> None:
    out_path = master_dir / f"combined{kind}MeanSimulatino{master_label}_Log.csv"
    rows: list[dict[str, str]] = []
    for run_num in run_nums:
        in_path = results_dir / str(run_num) / f"parsed{kind}MeanSimulatino{run_num}_Log.csv"
        if not in_path.exists():
            continue
        with open(in_path, newline="") as in_handle:
            reader = csv.DictReader(in_handle)
            for row in reader:
                if not row:
                    continue
                rows.append({name: row.get(name, "") for name in fieldnames})

    def _evo_key(row: dict[str, str]) -> float:
        try:
            return float(row.get("evolution rate", ""))
        except Exception:
            return float("inf")

    rows.sort(key=_evo_key)
    with open(out_path, "w", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _combine_master_means_for_update(
    results_dir: Path,
    master_dir: Path,
    master_label: str,
    run_nums: list[int],
) -> None:
    _combine_master_mean_kind_for_update(
        results_dir,
        master_dir,
        master_label,
        run_nums,
        "Arithmetic",
        [
            "evolution rate",
            "arithmetic mean length lived",
            "arithmetic mean species population time",
        ],
    )
    _combine_master_mean_kind_for_update(
        results_dir,
        master_dir,
        master_label,
        run_nums,
        "Geometric",
        [
            "evolution rate",
            "geometric mean length lived",
            "geometric mean species population time",
        ],
    )


def _rebuild_master_combined_for_update(
    results_dir: Path, master_dir: Path, run_nums: list[int]
) -> None:
    if not run_nums:
        return
    master_label = master_dir.name
    _combine_master_logs_for_update(results_dir, master_dir, master_label, run_nums)
    _combine_master_means_for_update(results_dir, master_dir, master_label, run_nums)


def _discover_env_run_nums(env_dir: Path) -> list[int]:
    out = []
    if not env_dir.is_dir():
        return out
    for entry in env_dir.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.isdigit():
            continue
        try:
            out.append(int(name))
        except Exception:
            continue
    return sorted(set(out))


def _latest_snapshot_points(run_dir: Path) -> list[tuple[float, float]]:
    snaps_dir = run_dir / "snapshots"
    if not snaps_dir.is_dir():
        return []
    newest = None
    newest_key = None
    for snap_path in snaps_dir.glob("arith_mean_*.csv"):
        if not snap_path.is_file():
            continue
        frame = None
        try:
            frame = int(snap_path.stem.rsplit("_", 1)[-1])
        except Exception:
            frame = None
        try:
            mtime = float(snap_path.stat().st_mtime)
        except Exception:
            mtime = 0.0
        key = (frame if frame is not None else -1, mtime, snap_path.name)
        if newest_key is None or key > newest_key:
            newest_key = key
            newest = snap_path
    if newest is None:
        return []
    return _extract_points_from_csv(newest)


def _snapshot_points_from_runs(env_dir: Path, run_nums: list[int]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for run_num in run_nums:
        run_dir = env_dir / str(int(run_num))
        merged.extend(_latest_snapshot_points(run_dir))
    return merged


def _predict_piecewise_gaussian(
    x: float,
    apex_x: float,
    apex_y: float,
    sigma_left: float,
    sigma_right: float,
    shape_power: float = 2.0,
) -> float:
    if apex_y <= 0:
        return 0.0
    sigma = sigma_left if x <= apex_x else sigma_right
    sigma = max(1e-9, sigma)
    power = max(1e-6, float(shape_power))
    dx = abs(float(x) - float(apex_x))
    expo = -((dx**power) / (2.0 * (sigma**power)))
    return apex_y * math.exp(expo)


def _format_desmos_power(value: float) -> str:
    if abs(float(value) - round(float(value))) < 1e-9:
        return str(int(round(float(value))))
    return f"{float(value):.6g}"


def _format_desmos_gaussian_expression(
    apex_y: float,
    apex_x: float,
    sigma: float,
    shape_power: float,
) -> str:
    power = _format_desmos_power(float(shape_power))
    center_sign = "-" if float(apex_x) >= 0 else "+"
    center_abs = abs(float(apex_x))
    return (
        f"{float(apex_y):.6g}*\\exp(-"
        f"(\\left(x{center_sign}{center_abs:.6g}\\right)^{{{power}}})"
        f"/(2*{float(sigma):.6g}^{{{power}}}))"
    )


def _format_desmos_piecewise_gaussian(
    apex_y: float,
    apex_x: float,
    sigma_left: float,
    sigma_right: float,
    shape_power: float,
) -> str:
    left_expr = _format_desmos_gaussian_expression(apex_y, apex_x, sigma_left, shape_power)
    right_expr = _format_desmos_gaussian_expression(apex_y, apex_x, sigma_right, shape_power)
    if abs(float(sigma_left) - float(sigma_right)) <= 1e-12:
        return left_expr
    return (
        f"\\left\\{{x\\le{float(apex_x):.6g}:{left_expr},"
        f"x>{float(apex_x):.6g}:{right_expr}\\right\\}}"
    )


def _fit_stitched_gaussian(points: list[tuple[float, float]]) -> dict | None:
    numeric_points: list[tuple[float, float]] = []
    for pair in points:
        if not isinstance(pair, (tuple, list)) or len(pair) < 2:
            continue
        x_val, y_val = pair[0], pair[1]
        if _is_number(x_val) and _is_number(y_val):
            numeric_points.append((float(x_val), float(y_val)))
    points = numeric_points
    if len(points) < 3:
        return None
    apex_point_x, apex_point_y = max(points, key=lambda p: p[1])
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    x_min = min(xs)
    x_max = max(xs)
    x_range = max(xs) - min(xs) if xs else 0.0
    if (not math.isfinite(x_range)) or x_range <= 0:
        return None

    sigma_min = max(1e-6, x_range * 1e-4)
    sigma_max = max(sigma_min * 10.0, x_range * 10.0)

    def _clamp_log_sigma(value: float) -> float:
        lo = math.log(sigma_min)
        hi = math.log(sigma_max)
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value

    # Keep master bell curves differentiable across the displayed evo-speed range.
    # Powers below 1 create cusp-like behavior at the apex; p=2 is the standard
    # Gaussian shape and has a smooth first derivative through the stitched apex.
    power_candidates = [2.0]

    def _clamp_apex(value: float) -> float:
        return max(float(x_min), min(float(x_max), float(value)))

    def _pearson_r_for_predicted(predicted: list[float]) -> float | None:
        if len(predicted) != len(ys) or len(predicted) < 2:
            return None
        mean_y = sum(ys) / float(len(ys))
        mean_pred = sum(predicted) / float(len(predicted))
        ss_y = sum((y - mean_y) ** 2 for y in ys)
        ss_pred = sum((p - mean_pred) ** 2 for p in predicted)
        if ss_y <= 1e-12 or ss_pred <= 1e-12:
            return None
        cov = sum((y - mean_y) * (p - mean_pred) for y, p in zip(ys, predicted))
        r_val = cov / math.sqrt(ss_y * ss_pred)
        return max(-1.0, min(1.0, float(r_val)))

    def _fit_is_better(candidate: dict | None, current: dict | None) -> bool:
        if not isinstance(candidate, dict):
            return False
        if not isinstance(current, dict):
            return True
        candidate_r = candidate.get("r")
        current_r = current.get("r")
        candidate_score = float(candidate_r) if _is_number(candidate_r) else -math.inf
        current_score = float(current_r) if _is_number(current_r) else -math.inf
        if candidate_score > current_score + 1e-10:
            return True
        if candidate_score < current_score - 1e-10:
            return False
        candidate_sse = float(candidate.get("sse", math.inf))
        current_sse = float(current.get("sse", math.inf))
        sse_eps = 1e-9 * max(1.0, abs(current_sse))
        if candidate_sse < current_sse - sse_eps:
            return True
        if candidate_sse > current_sse + sse_eps:
            return False
        candidate_apex = candidate.get("apex_x")
        current_apex = current.get("apex_x")
        if _is_number(candidate_apex) and _is_number(current_apex):
            return abs(float(candidate_apex) - float(apex_point_x)) < abs(
                float(current_apex) - float(apex_point_x)
            )
        return False

    def _solve_fit_stats(
        apex_x: float,
        sigma_left: float,
        sigma_right: float,
        shape_power: float,
    ) -> dict | None:
        apex_x = _clamp_apex(float(apex_x))
        sigma_left = max(sigma_min, min(sigma_max, float(sigma_left)))
        sigma_right = max(sigma_min, min(sigma_max, float(sigma_right)))
        power = max(0.1, float(shape_power))
        sum_yk = 0.0
        sum_k2 = 0.0
        ks: list[tuple[float, float]] = []
        for x, y in points:
            dx = abs(float(x) - float(apex_x))
            sigma = sigma_left if float(x) <= float(apex_x) else sigma_right
            k = math.exp(-((dx**power) / (2.0 * (sigma**power))))
            if not math.isfinite(k):
                continue
            sum_yk += float(y) * k
            sum_k2 += k * k
            ks.append((float(y), k))
        if sum_k2 <= 1e-15:
            return None
        amplitude = sum_yk / sum_k2
        if not math.isfinite(amplitude):
            return None
        sse = 0.0
        predicted = []
        for y, k in ks:
            pred = amplitude * k
            predicted.append(float(pred))
            diff = y - pred
            sse += diff * diff
        return {
            "apex_x": float(apex_x),
            "apex_y": float(amplitude),
            "sigma_left": float(sigma_left),
            "sigma_right": float(sigma_right),
            "shape_power": float(power),
            "sse": float(sse),
            "r": _pearson_r_for_predicted(predicted),
        }

    def _initial_sigma(apex_x: float, left_side: bool) -> float:
        d2_values = [
            (abs(float(x) - float(apex_x)) ** 2)
            for x, y in points
            if (
                (float(x) <= float(apex_x) if left_side else float(x) >= float(apex_x))
                and _is_number(y)
                and float(y) > 0
            )
        ]
        if not d2_values:
            return max(0.05, x_range * 0.12)
        mean_abs = sum(math.sqrt(max(0.0, d2)) for d2 in d2_values) / len(d2_values)
        return max(sigma_min, min(sigma_max, mean_abs if mean_abs > 0 else x_range * 0.12))

    def _fit_sigmas_for_apex(apex_x: float, shape_power: float) -> dict | None:
        apex_x = _clamp_apex(apex_x)
        sigma_left = _initial_sigma(apex_x, left_side=True)
        sigma_right = _initial_sigma(apex_x, left_side=False)
        best = _solve_fit_stats(apex_x, sigma_left, sigma_right, shape_power)
        if not isinstance(best, dict):
            return None
        best_log_left = _clamp_log_sigma(math.log(max(sigma_min, sigma_left)))
        best_log_right = _clamp_log_sigma(math.log(max(sigma_min, sigma_right)))

        step = 1.0
        for _ in range(7):
            improved = False
            for side in ("left", "right"):
                base_log = best_log_left if side == "left" else best_log_right
                candidates = [
                    base_log - step,
                    base_log - (step * 0.5),
                    base_log,
                    base_log + (step * 0.5),
                    base_log + step,
                ]
                local_best = dict(best)
                local_log_left = best_log_left
                local_log_right = best_log_right
                for cand in candidates:
                    cand_log = _clamp_log_sigma(cand)
                    log_left = cand_log if side == "left" else best_log_left
                    log_right = best_log_right if side == "left" else cand_log
                    trial = _solve_fit_stats(
                        apex_x,
                        math.exp(log_left),
                        math.exp(log_right),
                        shape_power,
                    )
                    if _fit_is_better(trial, local_best):
                        local_best = trial
                        local_log_left = log_left
                        local_log_right = log_right
                if _fit_is_better(local_best, best):
                    best = local_best
                    best_log_left = local_log_left
                    best_log_right = local_log_right
                    improved = True
            if not improved:
                step *= 0.5
                if step < 1e-3:
                    break
        return best

    def _apex_candidates() -> list[float]:
        candidates: list[float] = []
        seen: set[int] = set()
        scale = max(1e-12, x_range)

        def _add(value: float) -> None:
            value = _clamp_apex(value)
            key = int(round((value - x_min) / scale * 1_000_000_000))
            if key in seen:
                return
            seen.add(key)
            candidates.append(float(value))

        _add(float(apex_point_x))
        unique_xs = sorted(set(xs))
        sample_count = 17
        for idx in range(sample_count):
            _add(float(x_min) + (x_range * float(idx) / float(sample_count - 1)))
        if len(unique_xs) <= sample_count:
            for value in unique_xs:
                _add(float(value))
        else:
            for idx in range(sample_count):
                src_idx = int(
                    round(
                        float(idx) * float(len(unique_xs) - 1) / float(sample_count - 1)
                    )
                )
                _add(float(unique_xs[src_idx]))
        for x_val, _ in sorted(points, key=lambda p: p[1], reverse=True)[:6]:
            _add(float(x_val))
        return candidates

    def _fit_rank_key(item: dict) -> tuple[float, float, float]:
        r_val = item.get("r")
        r_score = float(r_val) if _is_number(r_val) else -math.inf
        sse = float(item.get("sse", math.inf))
        apex = (
            float(item.get("apex_x"))
            if _is_number(item.get("apex_x"))
            else float(apex_point_x)
        )
        return (r_score, -sse, -abs(apex - float(apex_point_x)))

    seed_fits = []
    for shape_power in power_candidates:
        for candidate_apex_x in _apex_candidates():
            sigma_left = _initial_sigma(candidate_apex_x, left_side=True)
            sigma_right = _initial_sigma(candidate_apex_x, left_side=False)
            seed = _solve_fit_stats(candidate_apex_x, sigma_left, sigma_right, shape_power)
            if isinstance(seed, dict):
                seed_fits.append(seed)

    if not seed_fits:
        return None

    global_best = None
    seed_fits.sort(key=_fit_rank_key, reverse=True)
    for seed in seed_fits[:8]:
        best = _fit_sigmas_for_apex(float(seed["apex_x"]), float(seed["shape_power"]))
        if _fit_is_better(best, global_best):
            global_best = best

    if not isinstance(global_best, dict):
        return None

    best_log_left = _clamp_log_sigma(math.log(max(sigma_min, float(global_best["sigma_left"]))))
    best_log_right = _clamp_log_sigma(math.log(max(sigma_min, float(global_best["sigma_right"]))))
    apex_step = x_range / 8.0
    log_step = 0.75
    shape_power = float(global_best["shape_power"])
    for _ in range(16):
        improved = False
        for dimension in ("apex", "left", "right"):
            if dimension == "apex":
                base = float(global_best["apex_x"])
                candidates = [
                    base - apex_step,
                    base - (apex_step * 0.5),
                    base,
                    base + (apex_step * 0.5),
                    base + apex_step,
                ]
            else:
                base = best_log_left if dimension == "left" else best_log_right
                candidates = [
                    base - log_step,
                    base - (log_step * 0.5),
                    base,
                    base + (log_step * 0.5),
                    base + log_step,
                ]
            local_best = dict(global_best)
            local_log_left = best_log_left
            local_log_right = best_log_right
            for cand in candidates:
                if dimension == "apex":
                    trial_apex = _clamp_apex(cand)
                    log_left = best_log_left
                    log_right = best_log_right
                elif dimension == "left":
                    trial_apex = float(global_best["apex_x"])
                    log_left = _clamp_log_sigma(cand)
                    log_right = best_log_right
                else:
                    trial_apex = float(global_best["apex_x"])
                    log_left = best_log_left
                    log_right = _clamp_log_sigma(cand)
                trial = _solve_fit_stats(
                    trial_apex,
                    math.exp(log_left),
                    math.exp(log_right),
                    shape_power,
                )
                if _fit_is_better(trial, local_best):
                    local_best = trial
                    local_log_left = log_left
                    local_log_right = log_right
            if _fit_is_better(local_best, global_best):
                global_best = local_best
                best_log_left = local_log_left
                best_log_right = local_log_right
                improved = True
        if not improved:
            apex_step *= 0.5
            log_step *= 0.5
            if apex_step < (x_range * 1e-5) and log_step < 1e-3:
                break

    if not isinstance(global_best, dict):
        return None
    sigma_left = float(global_best["sigma_left"])
    sigma_right = float(global_best["sigma_right"])
    shape_power = float(global_best["shape_power"])
    apex_x = float(global_best["apex_x"])
    apex_y = float(global_best["apex_y"])
    if not math.isfinite(apex_y):
        return None

    y_mean = sum(ys) / len(ys)
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = 0.0
    for x, y in points:
        pred = _predict_piecewise_gaussian(
            x,
            apex_x,
            apex_y,
            sigma_left,
            sigma_right,
            shape_power=shape_power,
        )
        ss_res += (y - pred) ** 2
    r2 = None
    if ss_tot > 0:
        r2 = 1.0 - (ss_res / ss_tot)

    equation = (
        f"y={apex_y:.6g}*exp(-(|x-{apex_x:.6g}|^{shape_power:.6g})/(2*{sigma_left:.6g}^{shape_power:.6g})) "
        f"for x<={apex_x:.6g}; "
        f"y={apex_y:.6g}*exp(-(|x-{apex_x:.6g}|^{shape_power:.6g})/(2*{sigma_right:.6g}^{shape_power:.6g})) "
        f"for x>{apex_x:.6g}"
    )
    desmos_equation = _format_desmos_piecewise_gaussian(
        apex_y,
        float(apex_x),
        sigma_left,
        sigma_right,
        shape_power,
    )
    return {
        "apex_x": float(apex_x),
        "apex_y": float(apex_y),
        "apex_point_x": float(apex_point_x),
        "apex_point_y": float(apex_point_y),
        "sigma_left": float(sigma_left),
        "sigma_right": float(sigma_right),
        "shape_power": float(shape_power),
        "r": (
            float(global_best["r"])
            if _is_number(global_best.get("r"))
            else None
        ),
        "r2": (None if r2 is None else float(r2)),
        "fit_objective": "pearson_r",
        "equation": equation,
        "desmos_equation": desmos_equation,
    }


def _cleanup_root_hub_alias_symlinks(results_root: Path) -> int:
    """Remove legacy top-level aliases that point into hub-owned data."""
    try:
        root_resolved = results_root.resolve()
    except Exception:
        root_resolved = results_root
    if not results_root.is_dir():
        return 0

    removed = 0
    for path in results_root.iterdir():
        if not path.is_symlink():
            continue
        name = path.name
        if (not name.startswith("master_")) and (not name.isdigit()):
            continue
        try:
            target = path.resolve(strict=True)
        except Exception:
            # Broken alias: safe to remove.
            target = None

        should_remove = False
        if target is None:
            should_remove = True
        else:
            try:
                rel = target.relative_to(root_resolved)
            except Exception:
                rel = None
            if rel is not None and len(rel.parts) >= 2 and rel.parts[0].startswith("hub_"):
                should_remove = True
        if not should_remove:
            continue
        try:
            path.unlink()
            removed += 1
        except Exception:
            continue
    return removed


def _plot_hub_scatter(rows: list[dict], out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm as mpl_cm
        from matplotlib import colors as mpl_colors
    except Exception:
        return False
    usable = [r for r in rows if r.get("env_rate") is not None and r.get("apex_x") is not None and r.get("apex_y") is not None]
    if not usable:
        return False

    xs = [float(r["env_rate"]) for r in usable]
    ys = [float(r["apex_x"]) for r in usable]
    fits = [float(r["apex_y"]) for r in usable]
    fit_min = min(fits)
    fit_max = max(fits)
    
    if fit_max <= fit_min:
        norms = [0.5 for _ in fits]
    else:
        denom = max(1e-9, fit_max - fit_min)
        norms = [(v - fit_min) / denom for v in fits]

    # Bubble area scale in points^2.
    # Keep points visible for small samples while limiting overlap for dense samples.
    point_count = max(1, len(fits))
    size_min = 14.0
    size_max = max(36.0, min(220.0, 520.0 / math.sqrt(float(point_count))))
    sizes = [size_min + (n * (size_max - size_min)) for n in norms]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    if x_max <= x_min:
        x_pad = 0.05
    else:
        x_pad = (x_max - x_min) * 0.04
    if y_max <= y_min:
        y_pad = 0.05
    else:
        y_pad = (y_max - y_min) * 0.04

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    fig.canvas.draw()

    plot_x, plot_y, plot_sizes = xs, ys, sizes
    color_min = fit_min if fits else 0.0
    color_max = fit_max if fits else 1.0
    if color_max <= color_min:
        color_max = color_min + 1e-9
    norm = mpl_colors.Normalize(vmin=color_min, vmax=color_max)
    rgba = mpl_cm.viridis(norm(fits))
    # Fitness-dependent transparency: lower fitness is more transparent.
    for idx, n in enumerate(norms):
        rgba[idx][3] = 0.25 + (0.75 * float(n))
    ax.scatter(
        plot_x,
        plot_y,
        s=plot_sizes,
        c=rgba,
        edgecolors="#0f172a",
        linewidths=0.65,
    )
    ax.set_title("Hub Apex Map: Environment Change Rate vs Evolution Speed")
    ax.set_xlabel("environment change rate")
    ax.set_ylabel("apex evolution speed")
    ax.grid(alpha=0.25)
    sm = mpl_cm.ScalarMappable(norm=norm, cmap=mpl_cm.viridis)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("fitness (apex arithmetic mean length lived)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True


def _plot_stitched_fits(rows: list[dict], out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except Exception:
        return False
    usable = [r for r in rows if r.get("fit") and r.get("points")]
    if not usable:
        return False

    usable = sorted(usable, key=lambda r: float(r["env_rate"]))
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    total = max(1, len(usable) - 1)
    for idx, row in enumerate(usable):
        rate = float(row["env_rate"])
        fit = row["fit"]
        points = row["points"]
        color = cm.viridis(idx / total)
        shape_power = (
            float(fit.get("shape_power"))
            if _is_number(fit.get("shape_power"))
            else 2.0
        )
        xs = sorted(set(float(x) for x, _ in points))
        if len(xs) < 2:
            continue
        min_x = xs[0]
        max_x = xs[-1]
        sample = []
        for i in range(100):
            x = min_x + ((max_x - min_x) * i / 99.0)
            y = _predict_piecewise_gaussian(
                x,
                float(fit["apex_x"]),
                float(fit["apex_y"]),
                float(fit["sigma_left"]),
                float(fit["sigma_right"]),
                shape_power=shape_power,
            )
            sample.append((x, y))
        ax.plot([p[0] for p in sample], [p[1] for p in sample], color=color, alpha=0.8, linewidth=1.2)
        ax.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            color=[color],
            alpha=0.12,
            s=8,
        )
        ax.text(float(fit["apex_x"]), float(fit["apex_y"]), f"{rate:.2f}", color=color, fontsize=7)
    ax.set_title("Per-Master Stitched Normal Fits (Apex-Locked)")
    ax.set_xlabel("evolution speed")
    ax.set_ylabel("fitness (arithmetic mean length lived)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True


def _plot_selected_master_view(row: dict, out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    if not isinstance(row, dict):
        return False
    raw_points = row.get("points")
    if not isinstance(raw_points, list):
        return False
    points = []
    for point in raw_points:
        if not isinstance(point, (tuple, list)) or len(point) < 2:
            continue
        evo_val = point[0]
        fit_val = point[1]
        if _is_number(evo_val) and _is_number(fit_val):
            points.append((float(evo_val), float(fit_val)))
    if not points:
        return False

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys, color="#4da6ff", alpha=0.35, s=18, edgecolors="none", label="run points")

    fit = row.get("fit")
    if isinstance(fit, dict):
        apex_x = fit.get("apex_x")
        apex_y = fit.get("apex_y")
        sigma_left = fit.get("sigma_left")
        sigma_right = fit.get("sigma_right")
        shape_power = (
            float(fit.get("shape_power"))
            if _is_number(fit.get("shape_power"))
            else 2.0
        )
        if (
            _is_number(apex_x)
            and _is_number(apex_y)
            and _is_number(sigma_left)
            and _is_number(sigma_right)
        ):
            min_x = min(xs)
            max_x = max(xs)
            if max_x <= min_x:
                min_x -= 0.05
                max_x += 0.05
            sample_x = []
            sample_y = []
            for idx in range(180):
                x_val = min_x + ((max_x - min_x) * float(idx) / 179.0)
                y_val = _predict_piecewise_gaussian(
                    float(x_val),
                    float(apex_x),
                    float(apex_y),
                    float(sigma_left),
                    float(sigma_right),
                    shape_power=shape_power,
                )
                if _is_number(y_val):
                    sample_x.append(float(x_val))
                    sample_y.append(float(y_val))
            if len(sample_x) >= 2:
                ax.plot(sample_x, sample_y, color="#f8c45b", linewidth=2.0, label="stitched fit")
            ax.scatter(
                [float(apex_x)],
                [float(apex_y)],
                color="#ffd166",
                edgecolors="#111111",
                linewidths=0.6,
                s=32,
                zorder=5,
            )

    master_run = row.get("master_run_num")
    env_rate = row.get("env_rate")
    title_bits = ["Selected Master Graph: evo speed vs fitness"]
    if _is_number(master_run):
        title_bits.append(f"master_{int(master_run)}")
    if _is_number(env_rate):
        title_bits.append(f"env={float(env_rate):.4g}")
    ax.set_title(" | ".join(title_bits))
    ax.set_xlabel("evo speed")
    ax.set_ylabel("fitness (arithmetic mean length lived)")
    ax.grid(alpha=0.25)
    try:
        ax.legend(loc="best")
    except Exception:
        pass
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True


def _plot_hub_full_view(rows: list[dict], out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm as mpl_cm
        from matplotlib import colors as mpl_colors
    except Exception:
        return False

    graph_points = []
    for row in rows:
        env_rate = row.get("env_rate")
        points = row.get("points")
        if (not _is_number(env_rate)) or (not isinstance(points, list)):
            continue
        env = float(env_rate)
        for point in points:
            if not isinstance(point, (tuple, list)) or len(point) < 2:
                continue
            evo_val = point[0]
            fit_val = point[1]
            if _is_number(evo_val) and _is_number(fit_val):
                graph_points.append(
                    {
                        "x": env,
                        "y": float(evo_val),
                        "fitness": float(fit_val),
                        "actual": True,
                    }
                )
    if not graph_points:
        return False

    xs = [float(p["x"]) for p in graph_points]
    ys = [float(p["y"]) for p in graph_points]
    fits = [float(p["fitness"]) for p in graph_points]

    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_pad = 0.05 if x_max <= x_min else (x_max - x_min) * 0.04
    y_pad = 0.05 if y_max <= y_min else (y_max - y_min) * 0.04

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    fit_min = min(fits)
    fit_max = max(fits)
    if fit_max <= fit_min:
        fit_max = fit_min + 1e-9
    norm = mpl_colors.Normalize(vmin=fit_min, vmax=fit_max)
    rgba = mpl_cm.viridis(norm(fits))
    for idx, fit_val in enumerate(fits):
        n = 0.5 if fit_max <= fit_min else (float(fit_val) - fit_min) / max(1e-9, fit_max - fit_min)
        rgba[idx][3] = 0.25 + (0.75 * max(0.0, min(1.0, n)))
    ax.scatter(xs, ys, s=14, c=rgba, edgecolors="#0f172a", linewidths=0.35)

    fit_report = _fit_hub_models_from_graph_points(graph_points)
    best_fit = fit_report.get("best_model") if isinstance(fit_report, dict) else None
    if isinstance(best_fit, dict):
        sample_x = []
        sample_y = []
        for idx in range(220):
            x_val = (x_min - x_pad) + (((x_max - x_min) + (2.0 * x_pad)) * float(idx) / 219.0)
            y_val = _eval_hub_model(best_fit, x_val)
            if _is_number(y_val):
                sample_x.append(float(x_val))
                sample_y.append(float(y_val))
        if len(sample_x) >= 2:
            ax.plot(sample_x, sample_y, color="#f8c45b", linewidth=2.0, alpha=0.9)

    ax.set_title("Full Hub View: env change rate vs evo speed (all masters)")
    ax.set_xlabel("env change rate")
    ax.set_ylabel("evo speed")
    ax.grid(alpha=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True


def _plot_ratio_curve(rows: list[dict], out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    usable = []
    for row in rows:
        rate = row.get("env_rate")
        apex_x = row.get("apex_x")
        apex_y = row.get("apex_y")
        if rate is None or apex_x is None or apex_y is None:
            continue
        if float(apex_x) <= 0:
            continue
        usable.append((float(rate), float(apex_y) / float(apex_x)))
    if not usable:
        return False
    usable.sort(key=lambda v: v[0])
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.plot([r for r, _ in usable], [ratio for _, ratio in usable], color="#cc3d1f", linewidth=2.0)
    ax.scatter([r for r, _ in usable], [ratio for _, ratio in usable], color="#f1a208", s=28)
    ax.set_title("Fitness / Evolution-Speed Ratio by Environment Change Rate")
    ax.set_xlabel("environment change rate")
    ax.set_ylabel("fitness / evolution speed")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return True


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _fmt_duration(seconds: float | None) -> str:
    if not _is_number(seconds):
        return "--:--"
    total = max(0, int(round(float(seconds))))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_eta(unix_ts: float | None) -> str:
    if not _is_number(unix_ts):
        return "--:--"
    try:
        return datetime.fromtimestamp(float(unix_ts)).strftime("%H:%M:%S")
    except Exception:
        return "--:--"


def _fit_text(font, text: str, max_width: int) -> str:
    raw = str(text) if text is not None else ""
    if max_width <= 0:
        return ""
    if font.size(raw)[0] <= max_width:
        return raw
    ellipsis = "..."
    if font.size(ellipsis)[0] >= max_width:
        return ""
    lo = 0
    hi = len(raw)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = raw[:mid] + ellipsis
        if font.size(candidate)[0] <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return raw[:lo] + ellipsis


def _linear_fit(xs: list[float], ys: list[float]):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    n = float(len(xs))
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = (n * sxx) - (sx * sx)
    if abs(den) < 1e-12:
        return None
    slope = ((n * sxy) - (sx * sy)) / den
    intercept = (sy - (slope * sx)) / n
    return slope, intercept


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    if n <= 0 or len(matrix) != n:
        return None
    aug = []
    for row_i in range(n):
        row = matrix[row_i]
        if len(row) != n:
            return None
        aug.append([float(v) for v in row] + [float(vector[row_i])])

    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_val = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pivot_val
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < 1e-12:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def _eval_polynomial(coeffs: list[float], x: float) -> float | None:
    try:
        x_val = float(x)
    except Exception:
        return None
    total = 0.0
    x_pow = 1.0
    for coeff in coeffs:
        try:
            total += float(coeff) * x_pow
        except Exception:
            return None
        x_pow *= x_val
    return total if math.isfinite(total) else None


def _format_polynomial_equation(coeffs: list[float]) -> str:
    terms = []
    for power, coeff in enumerate(coeffs):
        try:
            c = float(coeff)
        except Exception:
            continue
        if (not math.isfinite(c)) or abs(c) < 1e-12:
            continue
        mag = abs(c)
        if power == 0:
            term = f"{mag:.6g}"
        elif power == 1:
            term = f"{mag:.6g}*x"
        else:
            term = f"{mag:.6g}*x^{power}"
        if not terms:
            terms.append(term if c >= 0 else f"-{term}")
        else:
            sign = "+" if c >= 0 else "-"
            terms.append(f" {sign} {term}")
    if not terms:
        return "y=0"
    return "y=" + "".join(terms)


def _weighted_r2(
    samples: list[tuple[float, float, float]],
    predict_fn,
) -> float | None:
    if len(samples) < 2:
        return None
    sum_w = 0.0
    sum_y = 0.0
    for _, y_val, w_val in samples:
        w = float(w_val)
        if w <= 0 or (not math.isfinite(w)):
            continue
        y = float(y_val)
        if not math.isfinite(y):
            continue
        sum_w += w
        sum_y += w * y
    if sum_w <= 0:
        return None
    mean_y = sum_y / sum_w
    ss_tot = 0.0
    ss_res = 0.0
    for x_val, y_val, w_val in samples:
        w = float(w_val)
        if w <= 0 or (not math.isfinite(w)):
            continue
        y = float(y_val)
        if not math.isfinite(y):
            continue
        pred = predict_fn(float(x_val))
        if pred is None:
            return None
        try:
            yp = float(pred)
        except Exception:
            return None
        if not math.isfinite(yp):
            return None
        diff = y - yp
        ss_res += w * (diff * diff)
        d_tot = y - mean_y
        ss_tot += w * (d_tot * d_tot)
    if ss_tot <= 1e-12:
        return None
    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2) if math.isfinite(r2) else None


def _fit_weighted_polynomial_model(
    samples: list[tuple[float, float, float]],
    degree: int,
    model_name: str,
) -> dict | None:
    if degree < 1 or len(samples) < (degree + 1):
        return None
    m = degree + 1
    matrix = [[0.0 for _ in range(m)] for _ in range(m)]
    vector = [0.0 for _ in range(m)]
    for x_raw, y_raw, w_raw in samples:
        x = float(x_raw)
        y = float(y_raw)
        w = float(w_raw)
        if (not math.isfinite(x)) or (not math.isfinite(y)) or (not math.isfinite(w)) or w <= 0:
            continue
        max_pow = 2 * degree
        x_pows = [1.0]
        for _ in range(max_pow):
            x_pows.append(x_pows[-1] * x)
        for r in range(m):
            for c in range(m):
                matrix[r][c] += w * x_pows[r + c]
            vector[r] += w * y * x_pows[r]
    coeffs = _solve_linear_system(matrix, vector)
    if not coeffs:
        return None

    def _predict(x_value: float) -> float | None:
        return _eval_polynomial(coeffs, x_value)

    r2 = _weighted_r2(samples, _predict)
    sum_w = sum(float(w) for _, _, w in samples if _is_number(w) and float(w) > 0)
    return {
        "model": str(model_name),
        "r2": r2,
        "equation": _format_polynomial_equation(coeffs),
        "point_count": int(len(samples)),
        "weighted_point_count": int(round(sum_w)),
        "params": {"coeffs": [float(c) for c in coeffs]},
    }


def _fit_weighted_log_model(samples: list[tuple[float, float, float]]) -> dict | None:
    usable = []
    for x_raw, y_raw, w_raw in samples:
        x = float(x_raw)
        y = float(y_raw)
        w = float(w_raw)
        if x <= 0 or w <= 0:
            continue
        if (not math.isfinite(x)) or (not math.isfinite(y)) or (not math.isfinite(w)):
            continue
        usable.append((math.log(x), y, w))
    if len(usable) < 2:
        return None
    linear = _fit_weighted_polynomial_model(usable, degree=1, model_name="log")
    if not isinstance(linear, dict):
        return None
    params = linear.get("params", {})
    coeffs = params.get("coeffs") if isinstance(params, dict) else None
    if not isinstance(coeffs, list) or len(coeffs) < 2:
        return None
    b = float(coeffs[0])
    a = float(coeffs[1])
    base_samples = [(x, y, w) for x, y, w in samples if x > 0 and w > 0]

    def _predict(x_value: float) -> float | None:
        if x_value <= 0:
            return None
        y_hat = (a * math.log(x_value)) + b
        return y_hat if math.isfinite(y_hat) else None

    r2 = _weighted_r2(base_samples, _predict)
    b_sign = "+" if b >= 0 else "-"
    equation = f"y={a:.6g}*ln(x) {b_sign} {abs(b):.6g}"
    sum_w = sum(float(w) for _, _, w in base_samples if _is_number(w) and float(w) > 0)
    return {
        "model": "log",
        "r2": r2,
        "equation": equation,
        "point_count": int(len(base_samples)),
        "weighted_point_count": int(round(sum_w)),
        "params": {"a": a, "b": b},
    }


def _fit_weighted_bell_model(samples: list[tuple[float, float, float]]) -> dict | None:
    usable = []
    for x_raw, y_raw, w_raw in samples:
        x = float(x_raw)
        y = float(y_raw)
        w = float(w_raw)
        if y <= 0 or w <= 0:
            continue
        if (not math.isfinite(x)) or (not math.isfinite(y)) or (not math.isfinite(w)):
            continue
        usable.append((x, y, w))
    if len(usable) < 3:
        return None
    transformed = [(x, math.log(y), w) for x, y, w in usable]
    quad = _fit_weighted_polynomial_model(transformed, degree=2, model_name="bell")
    if not isinstance(quad, dict):
        return None
    params = quad.get("params", {})
    coeffs = params.get("coeffs") if isinstance(params, dict) else None
    if not isinstance(coeffs, list) or len(coeffs) < 3:
        return None
    c0 = float(coeffs[0])
    c1 = float(coeffs[1])
    c2 = float(coeffs[2])
    if c2 >= -1e-12:
        return None
    sigma_sq = -1.0 / (2.0 * c2)
    if sigma_sq <= 0 or (not math.isfinite(sigma_sq)):
        return None
    sigma = math.sqrt(sigma_sq)
    mu = c1 * sigma_sq
    amp = math.exp(c0 + ((mu * mu) / (2.0 * sigma_sq)))
    if (not math.isfinite(amp)) or amp <= 0:
        return None

    def _predict(x_value: float) -> float | None:
        expo = -((x_value - mu) ** 2) / (2.0 * sigma_sq)
        y_hat = amp * math.exp(expo)
        return y_hat if math.isfinite(y_hat) else None

    r2 = _weighted_r2(usable, _predict)
    equation = f"y={amp:.6g}*exp(-((x-{mu:.6g})^2)/(2*{sigma:.6g}^2))"
    sum_w = sum(float(w) for _, _, w in usable if _is_number(w) and float(w) > 0)
    return {
        "model": "bell_curve",
        "r2": r2,
        "equation": equation,
        "point_count": int(len(usable)),
        "weighted_point_count": int(round(sum_w)),
        "params": {"amp": amp, "mu": mu, "sigma": sigma},
    }


def _weighted_hub_samples_from_rows(rows: list[dict]) -> list[tuple[float, float, float]]:
    samples: list[tuple[float, float, float]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        env_rate = row.get("env_rate")
        points = row.get("points")
        if (not _is_number(env_rate)) or (not isinstance(points, list)):
            continue
        x = float(env_rate)
        for point in points:
            if not isinstance(point, (tuple, list)) or len(point) < 2:
                continue
            evo_val = point[0]
            fit_val = point[1]
            if not (_is_number(evo_val) and _is_number(fit_val)):
                continue
            weight = _fitness_weight_count(float(fit_val))
            if weight <= 0:
                continue
            samples.append((x, float(evo_val), float(weight)))
    return samples


def _weighted_hub_samples_from_graph_points(
    graph_points: list[dict],
) -> list[tuple[float, float, float]]:
    samples: list[tuple[float, float, float]] = []
    for point in graph_points:
        if not isinstance(point, dict):
            continue
        x_val = point.get("x")
        y_val = point.get("y")
        fit_val = point.get("fitness")
        if not (_is_number(x_val) and _is_number(y_val) and _is_number(fit_val)):
            continue
        weight = _fitness_weight_count(float(fit_val))
        if weight <= 0:
            continue
        samples.append((float(x_val), float(y_val), float(weight)))
    return samples


def _fit_hub_models_from_weighted_samples(
    samples: list[tuple[float, float, float]],
) -> dict:
    models = []
    linear = _fit_weighted_polynomial_model(samples, degree=1, model_name="linear")
    if isinstance(linear, dict):
        models.append(linear)
    quadratic = _fit_weighted_polynomial_model(samples, degree=2, model_name="quadratic")
    if isinstance(quadratic, dict):
        models.append(quadratic)
    polynomial = _fit_weighted_polynomial_model(samples, degree=3, model_name="polynomial")
    if isinstance(polynomial, dict):
        models.append(polynomial)
    log_model = _fit_weighted_log_model(samples)
    if isinstance(log_model, dict):
        models.append(log_model)
    bell_model = _fit_weighted_bell_model(samples)
    if isinstance(bell_model, dict):
        models.append(bell_model)
    valid = [
        model
        for model in models
        if _is_number(model.get("r2"))
    ]
    best_model = max(valid, key=lambda item: float(item["r2"])) if valid else None
    total_weighted = int(round(sum(float(w) for _, _, w in samples if _is_number(w) and float(w) > 0)))
    return {
        "selection_rule": "highest_r2",
        "point_count": int(len(samples)),
        "weighted_point_count": int(total_weighted),
        "models": models,
        "best_model": best_model,
    }


def _fit_hub_models_from_rows(rows: list[dict]) -> dict:
    return _fit_hub_models_from_weighted_samples(_weighted_hub_samples_from_rows(rows))


def _fit_hub_models_from_graph_points(graph_points: list[dict]) -> dict:
    return _fit_hub_models_from_weighted_samples(_weighted_hub_samples_from_graph_points(graph_points))


def _eval_hub_model(model: dict, x_value: float) -> float | None:
    if not isinstance(model, dict) or (not _is_number(x_value)):
        return None
    model_name = str(model.get("model", ""))
    params = model.get("params", {})
    if not isinstance(params, dict):
        params = {}
    x = float(x_value)
    if model_name in ("linear", "quadratic", "polynomial"):
        coeffs = params.get("coeffs")
        if not isinstance(coeffs, list):
            return None
        return _eval_polynomial([float(c) for c in coeffs], x)
    if model_name == "log":
        if x <= 0:
            return None
        a = params.get("a")
        b = params.get("b")
        if not (_is_number(a) and _is_number(b)):
            return None
        y = (float(a) * math.log(x)) + float(b)
        return y if math.isfinite(y) else None
    if model_name == "bell_curve":
        amp = params.get("amp")
        mu = params.get("mu")
        sigma = params.get("sigma")
        if not (_is_number(amp) and _is_number(mu) and _is_number(sigma)):
            return None
        s = float(sigma)
        if s <= 0:
            return None
        expo = -((x - float(mu)) ** 2) / (2.0 * s * s)
        y = float(amp) * math.exp(expo)
        return y if math.isfinite(y) else None
    return None


def _write_hub_stats_csv(path: Path, rows: list[dict]) -> dict:
    report = _fit_hub_models_from_rows(rows)
    best = report.get("best_model")
    best_model_name = str(best.get("model")) if isinstance(best, dict) else ""
    best_r2 = best.get("r2") if isinstance(best, dict) else None
    best_eq = str(best.get("equation", "")) if isinstance(best, dict) else ""
    try:
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "selected",
                    "selection rule",
                    "model",
                    "r2",
                    "equation",
                    "points used",
                    "weighted copies used",
                    "total points",
                    "total weighted copies",
                    "best model",
                    "best r2",
                    "best equation",
                ]
            )
            models = report.get("models", [])
            if isinstance(models, list):
                for model in models:
                    if not isinstance(model, dict):
                        continue
                    model_name = str(model.get("model", ""))
                    writer.writerow(
                        [
                            ("yes" if model_name == best_model_name else ""),
                            str(report.get("selection_rule", "highest_r2")),
                            model_name,
                            (
                                float(model["r2"])
                                if _is_number(model.get("r2"))
                                else ""
                            ),
                            str(model.get("equation", "")),
                            (
                                int(model["point_count"])
                                if isinstance(model.get("point_count"), int)
                                else ""
                            ),
                            (
                                int(model["weighted_point_count"])
                                if isinstance(model.get("weighted_point_count"), int)
                                else ""
                            ),
                            int(report.get("point_count", 0)),
                            int(report.get("weighted_point_count", 0)),
                            best_model_name,
                            (float(best_r2) if _is_number(best_r2) else ""),
                            best_eq,
                        ]
                    )
    except Exception:
        pass
    return report


def _project_metric(rows: list[dict], value_key: str) -> list[dict]:
    actual_points = []
    for row in rows:
        x = row.get("env_rate")
        y = row.get(value_key)
        if _is_number(x) and _is_number(y):
            actual_points.append((float(x), float(y)))

    model = None
    if len(actual_points) >= 2:
        model = _linear_fit([p[0] for p in actual_points], [p[1] for p in actual_points])
    elif len(actual_points) == 1:
        model = (0.0, float(actual_points[0][1]))

    projected = []
    indexed_rows = list(enumerate(rows))
    indexed_rows.sort(
        key=lambda item: (
            float(item[1].get("env_rate", 0.0))
            if _is_number(item[1].get("env_rate"))
            else 0.0
        )
    )
    for row_idx, row in indexed_rows:
        x = row.get("env_rate")
        if not _is_number(x):
            continue
        x = float(x)
        y_actual = row.get(value_key)
        point_meta = {
            "row_index": int(row_idx),
            "step_index": (
                int(row.get("step_index"))
                if isinstance(row.get("step_index"), int)
                else row.get("step_index")
            ),
            "status": row.get("status"),
            "master_run_num": row.get("master_run_num"),
            "planned_master_run_num": row.get("planned_master_run_num"),
            "max_species": row.get("max_species"),
            "total_species": row.get("total_species"),
            "max_frames": row.get("max_frames"),
            "duration_s": row.get("duration_s"),
        }
        if _is_number(y_actual):
            projected.append(
                {
                    "x": x,
                    "y": float(y_actual),
                    "actual": True,
                    **point_meta,
                }
            )
            continue
        y_pred = None
        if model is not None:
            y_pred = (model[0] * x) + model[1]
            if not _is_number(y_pred):
                y_pred = None
            elif y_pred < 0:
                y_pred = 0.0
        projected.append(
            {
                "x": x,
                "y": y_pred,
                "actual": False,
                **point_meta,
            }
        )
    return projected


def _project_final_bubble(rows: list[dict]) -> list[dict]:
    evo_points = _project_metric(rows, "apex_evolution_rate")
    fit_points = _project_metric(rows, "apex_fitness")
    fit_by_x = {float(p["x"]): p for p in fit_points if _is_number(p.get("x"))}

    out = []
    for evo in evo_points:
        if not _is_number(evo.get("x")) or not _is_number(evo.get("y")):
            continue
        x = float(evo["x"])
        fit = fit_by_x.get(x)
        fitness = None
        fit_actual = False
        if isinstance(fit, dict) and _is_number(fit.get("y")):
            fitness = float(fit["y"])
            fit_actual = bool(fit.get("actual"))
        out.append(
            {
                "x": x,
                "evo": float(evo["y"]),
                "fitness": fitness,
                "actual_evo": bool(evo.get("actual")),
                "actual_fitness": fit_actual,
                "actual": bool(evo.get("actual")) and fit_actual,
                "row_index": evo.get("row_index"),
                "step_index": evo.get("step_index"),
                "status": evo.get("status"),
                "master_run_num": evo.get("master_run_num"),
                "planned_master_run_num": evo.get("planned_master_run_num"),
                "max_species": evo.get("max_species"),
                "total_species": evo.get("total_species"),
                "max_frames": evo.get("max_frames"),
                "duration_s": evo.get("duration_s"),
            }
        )
    return out


def _runtime_prediction(rows: list[dict], start_ts: float, now_ts: float):
    elapsed = max(0.0, float(now_ts - start_ts))
    done = 0
    durations = []
    running_elapsed = None
    for row in rows:
        status = str(row.get("status", "pending"))
        if status in ("ok", "failed", "no_master", "stopped", "aborted"):
            done += 1
            dur = row.get("duration_s")
            if _is_number(dur):
                durations.append(float(dur))
        elif status == "running":
            start = row.get("started_at")
            if _is_number(start):
                running_elapsed = max(0.0, now_ts - float(start))
    avg = None
    if durations:
        avg = sum(durations) / len(durations)
    elif _is_number(running_elapsed):
        avg = float(running_elapsed)
    total = len(rows)
    remaining_steps = max(0, total - done)
    predicted_total = None
    remaining_seconds = None
    eta = None
    if _is_number(avg):
        predicted_total = float(avg) * float(total)
        remaining_seconds = max(0.0, predicted_total - elapsed)
        eta = now_ts + remaining_seconds
    return {
        "elapsed_s": elapsed,
        "remaining_s": remaining_seconds,
        "predicted_total_s": predicted_total,
        "eta_ts": eta,
        "done_steps": done,
        "total_steps": total,
    }


_FPS_MODE_CAPPED = 0
_FPS_MODE_UNCAPPED = 1
_FPS_MODE_FULL_THROTTLE = 2


class _HubDashboard:
    def __init__(
        self,
        enabled: bool,
        hub_idx: int,
        hub_dir: Path,
        planned_master_span: str,
        species_threshold: int,
        update_callback=None,
        reopen_callback=None,
        close_callback=None,
        viewer_callback=None,
        shutdown_callback=None,
    ) -> None:
        self.enabled = False
        self.closed = False
        self.scroll = 0
        self.scroll_target = 0.0
        self._last_draw = 0.0
        self._refresh_s = 0.1
        self.hub_idx = int(hub_idx)
        self.hub_dir = str(hub_dir)
        self.planned_master_span = str(planned_master_span)
        self.species_threshold = int(species_threshold)
        self.fps_mode = _FPS_MODE_CAPPED
        self.capped_fps = 1
        self.uncapped_fps = 120
        self.draw_fps = self.capped_fps
        self._click_uncap_duration_s = 3.0
        self._click_uncap_until = 0.0
        self.update_callback = update_callback
        self.reopen_callback = reopen_callback
        self.close_callback = close_callback
        self.viewer_callback = viewer_callback
        self.shutdown_callback = shutdown_callback
        self.selected_row_index = 0
        self._rows_cache = []
        self._table_rect = None
        self._row_h = 20
        self._rows_base_y = 0
        self._scroll_i = 0
        self._visible_rows = 0
        self._row_hitboxes = []
        self._pending_manual_update = False
        self._graph_plot_rect = None
        self._graph_dots = []
        self._selected_graph_point = None
        self._hub_fit_cache_key = None
        self._hub_fit_cache_report = None
        self._update_button_rect = None
        self._reopen_button_rect = None
        self._close_button_rect = None
        self._viewer_button_rect = None
        self._shutdown_button_rect = None
        self._equation_copy_hits = []
        self._copy_status_text = ""
        self._copy_status_ts = 0.0
        if not enabled:
            return
        try:
            import pygame  # pylint: disable=import-outside-toplevel
        except Exception:
            return
        self.pg = pygame
        try:
            pygame.init()
            self.window_w = 1360
            self.window_h = 820
            self.screen = pygame.display.set_mode((self.window_w, self.window_h))
            pygame.display.set_caption("Hub Runner Dashboard")
            self.font = pygame.font.SysFont("Consolas", 18)
            self.small = pygame.font.SysFont("Consolas", 15)
            self.title = pygame.font.SysFont("Consolas", 24)
            self.clock = pygame.time.Clock()
            self.enabled = True
        except Exception:
            self.enabled = False

    def _apply_fps_mode(self, new_mode: int) -> None:
        self.fps_mode = int(new_mode) % 3
        self._click_uncap_until = 0.0
        if self.fps_mode == _FPS_MODE_CAPPED:
            self._refresh_s = 0.1
            self.draw_fps = self.capped_fps
        elif self.fps_mode == _FPS_MODE_UNCAPPED:
            self._refresh_s = 0.0
            self.draw_fps = self.uncapped_fps
        else:
            self._refresh_s = 0.0
            self.draw_fps = 0

    def _click_uncap_active(self, now_ts: float | None = None) -> bool:
        if self.fps_mode != _FPS_MODE_CAPPED:
            return False
        if now_ts is None:
            now_ts = time.time()
        return float(now_ts) < float(self._click_uncap_until)

    def _arm_click_uncap(self) -> None:
        if self.fps_mode != _FPS_MODE_CAPPED:
            return
        self._click_uncap_until = max(
            float(self._click_uncap_until),
            float(time.time()) + float(self._click_uncap_duration_s),
        )

    def poll_sleep_seconds(self) -> float:
        if self.fps_mode == _FPS_MODE_FULL_THROTTLE:
            return 0.0
        if self.fps_mode == _FPS_MODE_UNCAPPED:
            return 0.02
        if self._click_uncap_active():
            return 0.02
        return 0.12

    def _set_scroll_target(self, value: float, max_scroll: int) -> None:
        self.scroll_target = max(0.0, min(float(max_scroll), float(value)))

    def _scroll_by(self, delta: float, max_scroll: int) -> None:
        self._set_scroll_target(self.scroll_target + float(delta), max_scroll)

    def _selected_row(self):
        if not self._rows_cache:
            return None
        idx = max(0, min(int(self.selected_row_index), len(self._rows_cache) - 1))
        self.selected_row_index = idx
        return self._rows_cache[idx]

    def _ensure_selected_visible(self) -> None:
        total = len(self._rows_cache)
        if total <= 0:
            return
        idx = max(0, min(int(self.selected_row_index), total - 1))
        self.selected_row_index = idx
        visible = max(1, int(self._visible_rows))
        max_scroll = max(0, total - visible)
        start_idx = int(max(0, min(max_scroll, int(self.scroll_target))))
        if idx < start_idx:
            self._set_scroll_target(float(idx), max_scroll)
        elif idx >= (start_idx + visible):
            self._set_scroll_target(float(idx - visible + 1), max_scroll)

    def _move_selected_row(self, delta: int) -> None:
        total = len(self._rows_cache)
        if total <= 0:
            self.selected_row_index = 0
            return
        idx = max(0, min(int(self.selected_row_index) + int(delta), total - 1))
        self.selected_row_index = idx
        self._selected_graph_point = None
        self._ensure_selected_visible()

    def _pick_graph_dot(self, mx: int, my: int):
        best = None
        best_d2 = None
        for entry in self._graph_dots:
            px = int(entry.get("px", -10_000))
            py = int(entry.get("py", -10_000))
            radius = int(entry.get("radius", 4))
            hit_r = max(6, radius + 4)
            dx = mx - px
            dy = my - py
            d2 = (dx * dx) + (dy * dy)
            if d2 > (hit_r * hit_r):
                continue
            if best_d2 is None or d2 < best_d2:
                best = entry
                best_d2 = d2
        if best is None:
            return None
        return best.get("point")

    def _row_index_from_graph_x(self, point: dict):
        if not isinstance(point, dict):
            return None
        target_x = point.get("x")
        if not _is_number(target_x):
            return None
        best_idx = None
        best_delta = None
        for idx, row in enumerate(self._rows_cache):
            row_x = row.get("env_rate")
            if not _is_number(row_x):
                continue
            delta = abs(float(row_x) - float(target_x))
            if best_delta is None or delta < best_delta:
                best_idx = int(idx)
                best_delta = delta
        return best_idx

    def _hub_fit_report(self, graph_points: list[dict]) -> dict:
        valid_count = 0
        sum_x = 0.0
        sum_y = 0.0
        sum_fit = 0.0
        sum_w = 0.0
        for point in graph_points:
            if not isinstance(point, dict):
                continue
            x_val = point.get("x")
            y_val = point.get("y")
            fit_val = point.get("fitness")
            if not (_is_number(x_val) and _is_number(y_val) and _is_number(fit_val)):
                continue
            weight = _fitness_weight_count(float(fit_val))
            if weight <= 0:
                continue
            valid_count += 1
            sum_x += float(x_val)
            sum_y += float(y_val)
            sum_fit += float(fit_val)
            sum_w += float(weight)
        signature = (
            int(len(graph_points)),
            int(valid_count),
            round(sum_x, 6),
            round(sum_y, 6),
            round(sum_fit, 6),
            round(sum_w, 3),
        )
        if signature == self._hub_fit_cache_key and isinstance(self._hub_fit_cache_report, dict):
            return self._hub_fit_cache_report
        report = _fit_hub_models_from_graph_points(graph_points)
        self._hub_fit_cache_key = signature
        self._hub_fit_cache_report = report
        return report

    def _trigger_reopen(self) -> None:
        if not callable(self.reopen_callback):
            return
        row = self._selected_row()
        if not isinstance(row, dict):
            return
        try:
            self.reopen_callback(row)
        except Exception:
            pass

    def _trigger_update(self) -> None:
        self._pending_manual_update = True
        # Force immediate redraw after a manual update request.
        self._last_draw = 0.0

    def _trigger_close(self) -> None:
        if not callable(self.close_callback):
            return
        row = self._selected_row()
        if not isinstance(row, dict):
            return
        try:
            self.close_callback(row)
        except Exception:
            pass

    def _trigger_viewer(self) -> None:
        if not callable(self.viewer_callback):
            return
        try:
            self.viewer_callback()
        except Exception:
            pass

    def _trigger_shutdown(self) -> None:
        if callable(self.shutdown_callback):
            try:
                self.shutdown_callback()
            except Exception:
                pass
        self.closed = True

    def _draw_equation_copy_button(self, rect: "pygame.Rect", equation_text: str, y: int) -> int:
        if not str(equation_text).strip():
            return 0
        label = "Copy"
        btn_w = max(40, int(self.small.size(label)[0]) + 10)
        btn_h = max(14, int(self.small.get_linesize()) - 1)
        btn_rect = self.pg.Rect(
            int(rect.right - btn_w - 6),
            int(y),
            int(btn_w),
            int(btn_h),
        )
        self.pg.draw.rect(self.screen, (52, 58, 76), btn_rect)
        self.pg.draw.rect(self.screen, (150, 156, 174), btn_rect, 1)
        txt = self.small.render(label, True, (232, 236, 245))
        self.screen.blit(
            txt,
            (
                btn_rect.x + (btn_rect.width - txt.get_width()) // 2,
                btn_rect.y + (btn_rect.height - txt.get_height()) // 2,
            ),
        )
        self._equation_copy_hits.append((btn_rect.copy(), str(equation_text)))
        return int(btn_rect.width + 8)

    def _handle_equation_copy_click(self, mx: int, my: int) -> bool:
        for item in reversed(self._equation_copy_hits):
            if (not isinstance(item, tuple)) or len(item) != 2:
                continue
            rect, eq_text = item
            if not isinstance(rect, self.pg.Rect):
                continue
            if not rect.collidepoint(mx, my):
                continue
            if _copy_to_clipboard(str(eq_text)):
                self._copy_status_text = "Equation copied to clipboard"
            else:
                self._copy_status_text = "Failed to copy equation to clipboard"
            self._copy_status_ts = float(time.time())
            return True
        return False

    def _pump_events(self) -> None:
        if not self.enabled:
            return
        for event in self.pg.event.get():
            if event.type == self.pg.QUIT:
                self.closed = True
            elif event.type == self.pg.KEYDOWN:
                if event.key in (self.pg.K_ESCAPE, self.pg.K_q):
                    self.closed = True
                elif event.key == self.pg.K_f:
                    self._apply_fps_mode((self.fps_mode + 1) % 3)
                elif event.key == self.pg.K_u:
                    self._trigger_update()
                elif event.key == self.pg.K_r:
                    self._trigger_reopen()
                elif event.key == self.pg.K_c:
                    self._trigger_close()
                elif event.key == self.pg.K_v:
                    self._trigger_viewer()
                elif event.key == self.pg.K_x:
                    self._trigger_shutdown()
                elif event.key == self.pg.K_UP:
                    self._move_selected_row(-1)
                elif event.key == self.pg.K_DOWN:
                    self._move_selected_row(1)
                elif event.key == self.pg.K_PAGEUP:
                    self.scroll_target = max(0.0, self.scroll_target - 12.0)
                elif event.key == self.pg.K_PAGEDOWN:
                    self.scroll_target += 12.0
                elif event.key == self.pg.K_HOME:
                    self.scroll_target = 0.0
                elif event.key == self.pg.K_END:
                    self.scroll_target = 1e9
            elif event.type == self.pg.MOUSEWHEEL:
                if event.y > 0:
                    self.scroll_target = max(0.0, self.scroll_target - 2.0)
                elif event.y < 0:
                    self.scroll_target += 2.0
            elif event.type == self.pg.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                self._arm_click_uncap()
                if self._handle_equation_copy_click(mx, my):
                    continue
                picked = self._pick_graph_dot(mx, my)
                if isinstance(picked, dict):
                    self._selected_graph_point = picked
                    row_idx = picked.get("row_index")
                    if isinstance(row_idx, int) and 0 <= row_idx < len(self._rows_cache):
                        self.selected_row_index = int(row_idx)
                    else:
                        by_x_idx = self._row_index_from_graph_x(picked)
                        if isinstance(by_x_idx, int) and 0 <= by_x_idx < len(self._rows_cache):
                            self.selected_row_index = int(by_x_idx)
                        else:
                            step_idx = picked.get("step_index")
                            if isinstance(step_idx, int) and 0 <= step_idx < len(self._rows_cache):
                                self.selected_row_index = int(step_idx)
                elif (
                    self._graph_plot_rect is not None
                    and self._graph_plot_rect.collidepoint(mx, my)
                ):
                    self._selected_graph_point = None
                if (
                    self._update_button_rect is not None
                    and self._update_button_rect.collidepoint(mx, my)
                ):
                    self._trigger_update()
                if (
                    self._reopen_button_rect is not None
                    and self._reopen_button_rect.collidepoint(mx, my)
                ):
                    self._trigger_reopen()
                if (
                    self._close_button_rect is not None
                    and self._close_button_rect.collidepoint(mx, my)
                ):
                    self._trigger_close()
                if (
                    self._viewer_button_rect is not None
                    and self._viewer_button_rect.collidepoint(mx, my)
                ):
                    self._trigger_viewer()
                if (
                    self._shutdown_button_rect is not None
                    and self._shutdown_button_rect.collidepoint(mx, my)
                ):
                    self._trigger_shutdown()
                if self._table_rect is not None and self._table_rect.collidepoint(mx, my):
                    picked_row = None
                    for hit in self._row_hitboxes:
                        if not isinstance(hit, tuple) or len(hit) != 3:
                            continue
                        row_idx, y0, y1 = hit
                        if y0 <= my < y1:
                            picked_row = int(row_idx)
                            break
                    if picked_row is not None and 0 <= picked_row < len(self._rows_cache):
                        self.selected_row_index = int(picked_row)
                    elif my >= self._rows_base_y:
                        local = int((my - self._rows_base_y) // self._row_h)
                        if 0 <= local < self._visible_rows:
                            idx = int(self._scroll_i + local)
                            if 0 <= idx < len(self._rows_cache):
                                self.selected_row_index = idx
        if self.closed:
            try:
                self.pg.display.quit()
                self.pg.quit()
            except Exception:
                pass
            self.enabled = False

    def _draw_master_inline_graph(self, row: dict, rect) -> None:
        pg = self.pg
        pg.draw.rect(self.screen, (13, 15, 20), rect)
        pg.draw.rect(self.screen, (58, 62, 72), rect, 1)
        title = self.small.render(
            _fit_text(self.small, "Selected Master Graph: evo speed vs fitness", rect.width - 12),
            True,
            (185, 195, 210),
        )
        self.screen.blit(title, (rect.x + 6, rect.y + 4))

        fit = row.get("fit")
        equation_text = str(fit.get("equation", "")).strip() if isinstance(fit, dict) else ""
        fit_lines = []
        if isinstance(fit, dict):
            apex_x = fit.get("apex_x")
            apex_y = fit.get("apex_y")
            sigma_left = fit.get("sigma_left")
            sigma_right = fit.get("sigma_right")
            shape_power = (
                float(fit.get("shape_power"))
                if _is_number(fit.get("shape_power"))
                else 2.0
            )
            r2 = fit.get("r2")
            if (
                _is_number(apex_x)
                and _is_number(apex_y)
                and _is_number(sigma_left)
                and _is_number(sigma_right)
            ):
                fit_lines.append(
                    "Best-fit generalized normal: y=A*exp(-(|x-mu|^p)/(2*sigma^p)); p fitted in [0.1,10]"
                )
                params = (
                    f"A={float(apex_y):.6g}, mu={float(apex_x):.6g}, "
                    f"sigmaL={float(sigma_left):.6g}, sigmaR={float(sigma_right):.6g}, p={float(shape_power):.6g}"
                )
                if _is_number(r2):
                    params += f", R^2={float(r2):.4f}"
                fit_lines.append(params)

        eq_font_size = max(8, int(round(float(self.small.get_height()) / 1.5)))
        eq_font = pg.font.SysFont("Consolas", eq_font_size)
        line_h = max(8, int(eq_font.get_linesize()))
        fit_base_y = rect.y + 18
        if equation_text:
            self._draw_equation_copy_button(rect, equation_text, rect.y + 4)
        if fit_lines:
            fit_w = max(0, rect.width - 12)
            for line_i, line_text in enumerate(fit_lines):
                fit_surf = eq_font.render(
                    _fit_text(eq_font, line_text, fit_w),
                    True,
                    (170, 182, 198),
                )
                self.screen.blit(fit_surf, (rect.x + 6, fit_base_y + (line_i * line_h)))

        points = []
        raw_points = row.get("points")
        if isinstance(raw_points, list):
            for point in raw_points:
                if not isinstance(point, (tuple, list)) or len(point) < 2:
                    continue
                evo_val = point[0]
                fit_val = point[1]
                if _is_number(evo_val) and _is_number(fit_val):
                    points.append((float(evo_val), float(fit_val)))

        plot_top = rect.y + 24 + (len(fit_lines) * line_h)
        plot = pg.Rect(rect.x + 34, plot_top, rect.width - 44, rect.height - ((plot_top - rect.y) + 8))
        if plot.width <= 20 or plot.height <= 20:
            return
        pg.draw.rect(self.screen, (9, 11, 15), plot)
        pg.draw.rect(self.screen, (52, 56, 64), plot, 1)

        if not points:
            msg = self.small.render(
                _fit_text(self.small, "No points yet for selected master.", plot.width - 8),
                True,
                (150, 150, 150),
            )
            self.screen.blit(msg, (plot.x + 6, plot.y + 6))
            return

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        raw_min_x = min(xs)
        raw_max_x = max(xs)
        raw_min_y = min(ys)
        raw_max_y = max(ys)
        if raw_max_x <= raw_min_x:
            min_x = raw_min_x - 0.05
            max_x = raw_max_x + 0.05
        else:
            x_pad = (raw_max_x - raw_min_x) * 0.04
            min_x = raw_min_x - x_pad
            max_x = raw_max_x + x_pad
        if raw_max_y <= raw_min_y:
            min_y = raw_min_y - 0.05
            max_y = raw_max_y + 0.05
        else:
            y_pad = (raw_max_y - raw_min_y) * 0.04
            min_y = raw_min_y - y_pad
            max_y = raw_max_y + y_pad

        def _to_px(xv: float, yv: float) -> tuple[int, int]:
            px = plot.x + int(((xv - min_x) / (max_x - min_x)) * plot.width)
            py = plot.y + plot.height - int(((yv - min_y) / (max_y - min_y)) * plot.height)
            return (px, py)

        fit_segments = []
        if isinstance(fit, dict):
            apex_x = fit.get("apex_x")
            apex_y = fit.get("apex_y")
            sigma_left = fit.get("sigma_left")
            sigma_right = fit.get("sigma_right")
            shape_power = (
                float(fit.get("shape_power"))
                if _is_number(fit.get("shape_power"))
                else 2.0
            )
            if (
                _is_number(apex_x)
                and _is_number(apex_y)
                and _is_number(sigma_left)
                and _is_number(sigma_right)
            ):
                sample = []
                for idx in range(120):
                    x_val = min_x + ((max_x - min_x) * float(idx) / 119.0)
                    y_val = _predict_piecewise_gaussian(
                        float(x_val),
                        float(apex_x),
                        float(apex_y),
                        float(sigma_left),
                        float(sigma_right),
                        shape_power=shape_power,
                    )
                    if _is_number(y_val):
                        sample.append((float(x_val), float(y_val)))
                if len(sample) >= 2:
                    for idx in range(1, len(sample)):
                        fit_segments.append((sample[idx - 1], sample[idx]))

        fit_min = min(ys)
        fit_max = max(ys)
        fit_denom = max(1e-9, fit_max - fit_min)
        for evo_val, fit_val in points:
            px, py = _to_px(evo_val, fit_val)
            n = max(0.0, min(1.0, (fit_val - fit_min) / fit_denom))
            radius = 1 + int(round(3 * n))
            color = (
                40 + int(210 * n),
                128 + int(95 * n),
                236 - int(166 * n),
            )
            pg.draw.circle(self.screen, color, (px, py), radius)
        for start, end in fit_segments:
            p0 = _to_px(start[0], start[1])
            p1 = _to_px(end[0], end[1])
            pg.draw.line(self.screen, (248, 196, 92), p0, p1, 2)

        min_y_txt = self.small.render(f"{raw_min_y:.3f}", True, (145, 145, 145))
        max_y_txt = self.small.render(f"{raw_max_y:.3f}", True, (145, 145, 145))
        min_x_txt = self.small.render(f"{raw_min_x:.3f}", True, (145, 145, 145))
        max_x_txt = self.small.render(f"{raw_max_x:.3f}", True, (145, 145, 145))
        x_lab = self.small.render("evo speed", True, (150, 150, 150))
        y_lab = self.small.render("fitness", True, (150, 150, 150))

        self.screen.blit(max_y_txt, (plot.x - 30, plot.y - 2))
        self.screen.blit(min_y_txt, (plot.x - 30, plot.bottom - 14))
        self.screen.blit(min_x_txt, (plot.x, plot.bottom - 14))
        self.screen.blit(max_x_txt, (plot.right - max_x_txt.get_width(), plot.bottom - 14))
        self.screen.blit(x_lab, (plot.x + 4, plot.y + 4))
        self.screen.blit(y_lab, (plot.x - 30, plot.y + 14))

    def _draw_graph(self, rect, rows: list[dict]) -> None:
        pg = self.pg
        self._graph_plot_rect = None
        self._graph_dots = []
        pg.draw.rect(self.screen, (22, 24, 29), rect)
        pg.draw.rect(self.screen, (70, 74, 86), rect, 1)
        graph_points = []
        for row in rows:
            env_rate = row.get("env_rate")
            points = row.get("points")
            if (not _is_number(env_rate)) or (not isinstance(points, list)):
                continue
            point_meta = {
                "graph_mode": "hub",
                "row_index": (
                    int(row.get("step_index"))
                    if isinstance(row.get("step_index"), int)
                    else row.get("step_index")
                ),
                "step_index": (
                    int(row.get("step_index"))
                    if isinstance(row.get("step_index"), int)
                    else row.get("step_index")
                ),
                "status": row.get("status"),
                "master_run_num": row.get("master_run_num"),
                "planned_master_run_num": row.get("planned_master_run_num"),
                "max_species": row.get("max_species"),
                "total_species": row.get("total_species"),
                "max_frames": row.get("max_frames"),
                "duration_s": row.get("duration_s"),
            }
            for src_idx, point in enumerate(points, start=1):
                if not isinstance(point, (tuple, list)) or len(point) < 2:
                    continue
                evo_val = point[0]
                fit_val = point[1]
                if not (_is_number(evo_val) and _is_number(fit_val)):
                    continue
                graph_points.append(
                    {
                        "x": float(env_rate),
                        "y": float(evo_val),
                        "evo": float(evo_val),
                        "fitness": float(fit_val),
                        "actual": True,
                        "source_row_index": int(src_idx),
                        **point_meta,
                    }
                )
        fit_report = self._hub_fit_report(graph_points)
        best_fit = fit_report.get("best_model") if isinstance(fit_report, dict) else None

        selected = self._selected_graph_point if isinstance(self._selected_graph_point, dict) else None
        if selected is not None and str(selected.get("graph_mode", "hub")) != "hub":
            selected = None
            self._selected_graph_point = None

        fit_font = pg.font.SysFont("Consolas", 12)
        fit_line_h = max(10, int(fit_font.get_linesize()))
        fit_line_gap = 2
        header_x = rect.x + 10

        def _wrap_fit_line(text: str, max_w: int) -> list[str]:
            raw = str(text).strip()
            if (not raw) or max_w <= 8:
                return [raw]
            lines = []
            remaining = raw
            while remaining:
                if fit_font.size(remaining)[0] <= max_w:
                    lines.append(remaining)
                    break
                cut = len(remaining)
                while cut > 1 and fit_font.size(remaining[:cut])[0] > max_w:
                    cut -= 1
                split = remaining.rfind(" ", 0, cut)
                if split <= 0:
                    split = cut
                chunk = remaining[:split].rstrip()
                if chunk:
                    lines.append(chunk)
                remaining = remaining[split:].lstrip()
            return lines if lines else [raw]

        header_lines = []
        copy_reserved = 0
        if isinstance(best_fit, dict) and _is_number(best_fit.get("r2")):
            equation = str(best_fit.get("equation", ""))
            copy_reserved = self._draw_equation_copy_button(rect, equation, rect.y + 8)
            header_w = max(0, rect.width - 20 - copy_reserved)
            header_lines.extend(_wrap_fit_line(f"Equation: {equation}", header_w))
            header_lines.extend(_wrap_fit_line(f"R^2: {float(best_fit['r2']):.4f}", header_w))
            fit_info_color = (234, 210, 147)
        else:
            header_w = max(0, rect.width - 20)
            header_lines.extend(_wrap_fit_line("Equation: not enough data", header_w))
            header_lines.extend(_wrap_fit_line("R^2: --", header_w))
            fit_info_color = (165, 165, 165)

        header_top = rect.y + 10
        if header_w >= 20:
            yy = header_top
            for line in header_lines:
                line_surf = fit_font.render(str(line), True, fit_info_color)
                self.screen.blit(line_surf, (header_x, yy))
                yy += fit_line_h + fit_line_gap
            header_bottom = yy
        else:
            header_bottom = header_top

        xs = [float(p["x"]) for p in graph_points if _is_number(p.get("x"))]
        ys = [float(p["y"]) for p in graph_points if _is_number(p.get("y"))]
        if not ys or not xs:
            msg = self.small.render(
                _fit_text(
                    self.small,
                    "Need started/completed master points to draw graph.",
                    rect.width - 24,
                ),
                True,
                (170, 170, 170),
            )
            self.screen.blit(msg, (rect.x + 12, int(header_bottom) + 8))
            return

        raw_min_x = min(xs)
        raw_max_x = max(xs)
        raw_min_y = min(ys)
        raw_max_y = max(ys)
        if raw_max_x <= raw_min_x:
            min_x = raw_min_x - 0.05
            max_x = raw_max_x + 0.05
        else:
            x_pad = (raw_max_x - raw_min_x) * 0.04
            min_x = raw_min_x - x_pad
            max_x = raw_max_x + x_pad
        if raw_max_y <= raw_min_y:
            min_y = raw_min_y - 0.05
            max_y = raw_max_y + 0.05
        else:
            y_pad = (raw_max_y - raw_min_y) * 0.04
            min_y = raw_min_y - y_pad
            max_y = raw_max_y + y_pad

        plot_top = int(header_bottom) + 8
        plot = pg.Rect(
            rect.x + 46,
            plot_top,
            rect.width - 62,
            rect.height - ((plot_top - rect.y) + 24),
        )
        self._graph_plot_rect = plot
        pg.draw.rect(self.screen, (16, 18, 23), plot)
        pg.draw.rect(self.screen, (64, 68, 79), plot, 1)

        def _to_px(xv, yv):
            px = plot.x + int(((xv - min_x) / (max_x - min_x)) * plot.width)
            py = plot.y + plot.height - int(((yv - min_y) / (max_y - min_y)) * plot.height)
            return px, py

        def _axis_fmt(value: float, span: float) -> str:
            if span >= 10:
                return f"{value:.1f}"
            if span >= 1:
                return f"{value:.2f}"
            return f"{value:.3f}"

        fits = [float(p["fitness"]) for p in graph_points if _is_number(p.get("fitness"))]
        fit_min = min(fits) if fits else 0.0
        fit_max = max(fits) if fits else 1.0
        fit_denom = max(1e-9, fit_max - fit_min)

        def _fit_norm(value):
            if not _is_number(value):
                return 0.35
            return max(0.0, min(1.0, (float(value) - fit_min) / fit_denom))

        fit_segments = []
        if isinstance(best_fit, dict):
            y_span = max(1e-9, max_y - min_y)
            sample = []
            for idx in range(160):
                x_val = min_x + ((max_x - min_x) * float(idx) / 159.0)
                y_val = _eval_hub_model(best_fit, x_val)
                if not _is_number(y_val):
                    continue
                y_float = float(y_val)
                if y_float < (min_y - (2.0 * y_span)) or y_float > (max_y + (2.0 * y_span)):
                    continue
                sample.append((x_val, y_float))
            if len(sample) >= 2:
                for idx in range(1, len(sample)):
                    fit_segments.append((sample[idx - 1], sample[idx]))

        selected_found = False
        for point in graph_points:
            px, py = _to_px(float(point["x"]), float(point["y"]))
            n = _fit_norm(point.get("fitness"))
            radius = max(1, int(round(4 * n)))
            is_selected = False
            selected = self._selected_graph_point
            if isinstance(selected, dict):
                sel_mode = str(selected.get("graph_mode", "hub"))
                cur_mode = str(point.get("graph_mode", "hub"))
                sel_step = selected.get("step_index")
                cur_step = point.get("step_index")
                sel_src = selected.get("source_row_index")
                cur_src = point.get("source_row_index")
                if (
                    sel_mode == cur_mode
                    and isinstance(sel_step, int)
                    and isinstance(cur_step, int)
                    and isinstance(sel_src, int)
                    and isinstance(cur_src, int)
                ):
                    is_selected = (int(sel_step) == int(cur_step)) and (int(sel_src) == int(cur_src))
                elif sel_mode == cur_mode and isinstance(sel_step, int) and isinstance(cur_step, int):
                    is_selected = int(sel_step) == int(cur_step)
                else:
                    try:
                        is_selected = (
                            abs(float(selected.get("x")) - float(point.get("x"))) <= 1e-9
                            and abs(float(selected.get("y")) - float(point.get("y"))) <= 1e-9
                        )
                    except Exception:
                        is_selected = False
            color = (
                35 + int(220 * n),
                125 + int(110 * n),
                235 - int(165 * n),
            )
            alpha = 70 + int(185 * n)
            dot = pg.Surface((radius * 2 + 2, radius * 2 + 2), pg.SRCALPHA)
            pg.draw.circle(
                dot,
                (int(color[0]), int(color[1]), int(color[2]), int(alpha)),
                (radius + 1, radius + 1),
                radius,
            )
            self.screen.blit(dot, (px - radius - 1, py - radius - 1))
            if is_selected:
                selected_found = True
                pg.draw.circle(self.screen, (255, 255, 255), (px, py), radius + 3, 1)
            self._graph_dots.append(
                {
                    "px": int(px),
                    "py": int(py),
                    "radius": int(radius),
                    "point": point,
                }
            )
        for start, end in fit_segments:
            p0 = _to_px(start[0], start[1])
            p1 = _to_px(end[0], end[1])
            pg.draw.line(self.screen, (248, 196, 92), p0, p1, 2)
        if self._selected_graph_point is not None and (not selected_found):
            self._selected_graph_point = None

        x_span = max(1e-9, raw_max_x - raw_min_x)
        y_span = max(1e-9, raw_max_y - raw_min_y)
        min_label = self.small.render(_axis_fmt(raw_min_y, y_span), True, (150, 150, 150))
        max_label = self.small.render(_axis_fmt(raw_max_y, y_span), True, (150, 150, 150))
        self.screen.blit(max_label, (plot.x - 40, plot.y - 6))
        self.screen.blit(min_label, (plot.x - 40, plot.bottom - 8))
        x1 = self.small.render(_axis_fmt(raw_min_x, x_span), True, (150, 150, 150))
        x2 = self.small.render(_axis_fmt(raw_max_x, x_span), True, (150, 150, 150))
        self.screen.blit(x1, (plot.x, plot.bottom + 4))
        self.screen.blit(x2, (plot.right - x2.get_width(), plot.bottom + 4))
        x_label = self.small.render("env change rate", True, (155, 155, 155))
        y_label = self.small.render("evo speed", True, (155, 155, 155))
        self.screen.blit(x_label, (plot.x + 4, plot.bottom + 22))
        self.screen.blit(y_label, (plot.x - 42, plot.y + 8))

    def update(self, state: dict, force: bool = False) -> None:
        if not self.enabled:
            return
        self._rows_cache = list(state.get("rows", []))
        if self._rows_cache:
            self.selected_row_index = max(
                0, min(int(self.selected_row_index), len(self._rows_cache) - 1)
            )
        else:
            self.selected_row_index = 0
        self._pump_events()
        if not self.enabled:
            return
        if self._pending_manual_update:
            self._pending_manual_update = False
            if callable(self.update_callback):
                try:
                    self.update_callback(self._selected_row())
                except TypeError:
                    try:
                        self.update_callback()
                    except Exception:
                        pass
                except Exception:
                    pass
            self._rows_cache = list(state.get("rows", []))
            force = True
        now = time.time()
        refresh_interval = 0.0 if self._click_uncap_active(now) else float(self._refresh_s)
        if (not force) and ((now - self._last_draw) < refresh_interval):
            return
        self._last_draw = now
        pg = self.pg
        self.screen.fill((12, 13, 17))
        self._equation_copy_hits = []

        rows = self._rows_cache
        pred = _runtime_prediction(rows, float(state.get("start_ts", now)), now)
        elapsed = _fmt_duration(pred.get("elapsed_s"))
        remain = _fmt_duration(pred.get("remaining_s"))
        total_pred = _fmt_duration(pred.get("predicted_total_s"))
        eta = _fmt_eta(pred.get("eta_ts"))
        completed = int(pred.get("done_steps", 0))
        total = int(pred.get("total_steps", 0))
        running_row = state.get("running_row")
        status = str(state.get("status", "running"))
        fps_mode_label = {
            _FPS_MODE_CAPPED: f"CAPPED({self.capped_fps})",
            _FPS_MODE_UNCAPPED: f"UNCAPPED({self.uncapped_fps})",
            _FPS_MODE_FULL_THROTTLE: "FULL",
        }.get(self.fps_mode, "CAPPED")
        if self._click_uncap_active(now):
            fps_mode_label = f"{fps_mode_label}+CLICK({int(self._click_uncap_duration_s)}s)"
        self._shutdown_button_rect = pg.Rect(self.window_w - 188, 96, 168, 30)
        header_left = 20
        header_max_w = max(160, self._shutdown_button_rect.x - header_left - 12)
        header1 = self.title.render(
            _fit_text(
                self.title,
                f"HUB {self.hub_idx} Dashboard    status: {status.upper()}",
                header_max_w,
            ),
            True,
            (230, 230, 230),
        )
        self.screen.blit(header1, (header_left, 16))
        header2 = self.font.render(
            _fit_text(
                self.font,
                f"Hub dir: {self.hub_dir}    planned master ids: {self.planned_master_span}",
                header_max_w,
            ),
            True,
            (180, 180, 180),
        )
        self.screen.blit(header2, (header_left, 48))
        header3 = self.font.render(
            _fit_text(
                self.font,
                f"Elapsed: {elapsed}    Pred total: {total_pred}    Remaining: {remain}    ETA: {eta}",
                header_max_w,
            ),
            True,
            (170, 220, 180),
        )
        self.screen.blit(header3, (header_left, 74))
        header4 = self.font.render(
            _fit_text(
                self.font,
                f"Completed: {completed}/{total}    Species threshold: {self.species_threshold}    Running step: {running_row if running_row is not None else '-'}    FPS mode: {fps_mode_label} (F)",
                header_max_w,
            ),
            True,
            (170, 200, 230),
        )
        self.screen.blit(header4, (header_left, 100))
        pg.draw.rect(self.screen, (110, 50, 50), self._shutdown_button_rect)
        pg.draw.rect(self.screen, (180, 160, 160), self._shutdown_button_rect, 1)
        close_hub_text = self.small.render("Close Hub (X)", True, (235, 235, 235))
        self.screen.blit(
            close_hub_text,
            (
                self._shutdown_button_rect.x
                + (self._shutdown_button_rect.width - close_hub_text.get_width()) // 2,
                self._shutdown_button_rect.y + 7,
            ),
        )

        left_margin = 20
        top_y = 132
        right_margin = 20
        mid_gap = 16
        min_table_w = 704
        min_right_w = 380
        available_w = self.window_w - left_margin - right_margin - mid_gap
        if available_w < (min_table_w + min_right_w):
            min_table_w = max(560, available_w - min_right_w)
            min_right_w = max(260, available_w - min_table_w)
        table_w = max(min_table_w, int(available_w * 0.67))
        table_w = min(table_w, max(min_table_w, available_w - min_right_w))
        right_w = max(min_right_w, available_w - table_w)
        right_x = self.window_w - right_margin - right_w
        table_h = max(320, self.window_h - top_y - 16)
        table_rect = pg.Rect(left_margin, top_y, table_w, table_h)
        self._table_rect = table_rect
        pg.draw.rect(self.screen, (18, 20, 25), table_rect)
        pg.draw.rect(self.screen, (70, 74, 86), table_rect, 1)
        cols = [
            ("step", 50),
            ("rate", 72),
            ("plan", 72),
            ("master", 72),
            ("status", 90),
            ("species", 112),
            ("frames", 90),
            ("dur", 66),
        ]
        col_layout = []
        x = table_rect.x + 8
        y = table_rect.y + 8
        for name, width in cols:
            col_layout.append((x, width))
            surf = self.small.render(
                _fit_text(self.small, name, max(8, width - 6)),
                True,
                (190, 190, 190),
            )
            self.screen.blit(surf, (x, y))
            x += width
        divider_y = y + 20
        pg.draw.line(
            self.screen,
            (58, 60, 70),
            (table_rect.x + 6, divider_y),
            (table_rect.right - 6, divider_y),
            1,
        )

        row_h = 20
        table_footer_h = 26
        base_y = divider_y + 6
        rows_bottom = table_rect.bottom - table_footer_h - 6
        available_rows_h = max(0, rows_bottom - base_y)
        max_rows = max(1, available_rows_h // row_h)
        total_rows = len(rows)
        max_scroll = max(0, total_rows - max_rows)
        self._set_scroll_target(self.scroll_target, max_scroll)
        self.scroll += (self.scroll_target - self.scroll) * 0.35
        if abs(self.scroll_target - self.scroll) < 0.05:
            self.scroll = self.scroll_target
        self.scroll = max(0.0, min(float(max_scroll), float(self.scroll)))
        # Keep sub-row precision so smooth scroll can still resolve to index 0.
        # Rounding here can trap at index 1 when target is 0.
        scroll_i = int(self.scroll)
        selected_idx = int(self.selected_row_index)
        selected_extra_h = 188
        visible_entries = []
        cursor_y = base_y
        row_cursor = scroll_i
        while row_cursor < total_rows and cursor_y < rows_bottom:
            extra_h = selected_extra_h if row_cursor == selected_idx else 0
            needed_h = row_h + extra_h
            if (cursor_y + needed_h) > rows_bottom:
                if not visible_entries:
                    visible_entries.append((row_cursor, cursor_y, extra_h))
                break
            visible_entries.append((row_cursor, cursor_y, extra_h))
            cursor_y += needed_h
            row_cursor += 1

        self._row_h = row_h
        self._rows_base_y = base_y
        self._scroll_i = scroll_i
        self._visible_rows = len(visible_entries)
        self._row_hitboxes = []

        for vis_i, entry in enumerate(visible_entries):
            row_idx, ry, extra_h = entry
            row = rows[row_idx]
            if vis_i % 2 == 0:
                pg.draw.rect(self.screen, (15, 17, 22), (table_rect.x + 4, ry - 1, table_rect.width - 8, row_h))
            if row_idx == int(self.selected_row_index):
                pg.draw.rect(
                    self.screen,
                    (34, 52, 72),
                    (table_rect.x + 3, ry - 1, table_rect.width - 6, row_h),
                )
            status_text = str(row.get("status", "pending"))
            color = {
                "ok": (150, 230, 160),
                "failed": (255, 140, 140),
                "running": (255, 220, 120),
                "pending": (170, 170, 170),
                "no_master": (255, 160, 120),
                "stopped": (255, 180, 120),
                "aborted": (255, 130, 130),
            }.get(status_text, (200, 200, 200))
            dur = row.get("duration_s")
            if status_text == "running" and _is_number(row.get("started_at")):
                base_dur = row.get("duration_base_s")
                if not _is_number(base_dur):
                    base_dur = 0.0
                dur = float(base_dur) + max(0.0, now - float(row["started_at"]))
            values = [
                str(int(row.get("step_index", 0)) + 1),
                f"{float(row.get('env_rate', 0.0)):.2f}" if _is_number(row.get("env_rate")) else "",
                "" if row.get("planned_master_run_num") is None else str(row.get("planned_master_run_num")),
                "" if row.get("master_run_num") is None else str(row.get("master_run_num")),
                status_text,
                (
                    f"{float(row.get('total_species')):.1f}"
                    if _is_number(row.get("total_species"))
                    else (
                        f"{float(row.get('max_species')):.1f}"
                        if _is_number(row.get("max_species"))
                        else ""
                    )
                ),
                f"{float(row.get('max_frames')):.0f}" if _is_number(row.get("max_frames")) else "",
                _fmt_duration(dur),
            ]
            for c_idx, (cell_x, width) in enumerate(col_layout):
                text = self.small.render(
                    _fit_text(self.small, values[c_idx], max(8, width - 6)),
                    True,
                    color if c_idx == 4 else (205, 205, 205),
                )
                self.screen.blit(text, (cell_x, ry))
            self._row_hitboxes.append((int(row_idx), int(ry), int(ry + row_h + max(0, extra_h))))

            if row_idx == int(self.selected_row_index) and extra_h > 0:
                details_rect = pg.Rect(
                    table_rect.x + 6,
                    ry + row_h,
                    table_rect.width - 12,
                    extra_h - 2,
                )
                pg.draw.rect(self.screen, (21, 30, 41), details_rect)
                pg.draw.rect(self.screen, (64, 79, 97), details_rect, 1)
                info_x = details_rect.x + 8
                info_y = details_rect.y + 6
                line_w = details_rect.width - 16
                master_val = row.get("master_run_num")
                master_text = f"master_{master_val}" if master_val is not None else "master: --"
                step_text = str(int(row.get("step_index", 0)) + 1)
                rate_text = (
                    f"{float(row.get('env_rate')):.2f}"
                    if _is_number(row.get("env_rate"))
                    else "--"
                )
                species_text = (
                    f"{float(row.get('total_species')):.1f}"
                    if _is_number(row.get("total_species"))
                    else (
                        f"{float(row.get('max_species')):.1f}"
                        if _is_number(row.get("max_species"))
                        else "--"
                    )
                )
                dur_val = row.get("duration_s")
                dur_text = _fmt_duration(float(dur_val)) if _is_number(dur_val) else "--:--"
                selected_lines = [
                    f"Selected: step {step_text} | {master_text} | status: {status_text} | env: {rate_text}",
                    f"Species: {species_text} | Duration: {dur_text}",
                ]
                for line in selected_lines:
                    surf = self.small.render(
                        _fit_text(self.small, line, line_w),
                        True,
                        (188, 202, 224),
                    )
                    self.screen.blit(surf, (info_x, info_y))
                    info_y += 18

                graph_rect = pg.Rect(
                    details_rect.x + 8,
                    info_y + 2,
                    details_rect.width - 16,
                    max(60, details_rect.bottom - (info_y + 10)),
                )
                self._draw_master_inline_graph(row, graph_rect)

        if visible_entries:
            rows_first = int(visible_entries[0][0]) + 1
            rows_last = int(visible_entries[-1][0]) + 1
        else:
            rows_first = 0
            rows_last = 0
        scroll_info = self.small.render(
            _fit_text(
                self.small,
                f"rows {rows_first}-{rows_last} / {total_rows} (up/down select, wheel/page/home/end scroll)",
                table_rect.width - 20,
            ),
            True,
            (150, 150, 150),
        )
        self.screen.blit(scroll_info, (table_rect.x + 8, table_rect.bottom - table_footer_h + 4))
        if total_rows > max_rows:
            bar_x = table_rect.right - 8
            bar_y = base_y
            bar_h = max(18, rows_bottom - base_y)
            self.pg.draw.rect(self.screen, (44, 47, 56), (bar_x, bar_y, 4, bar_h))
            thumb_h = max(18, int((max_rows / max(1, total_rows)) * bar_h))
            top_ratio = float(self.scroll) / max(1.0, float(max_scroll))
            thumb_y = bar_y + int((bar_h - thumb_h) * top_ratio)
            self.pg.draw.rect(self.screen, (120, 130, 150), (bar_x - 1, thumb_y, 6, thumb_h))

        right_x = table_rect.right + mid_gap
        right_w = max(320, self.window_w - right_x - right_margin)
        right_h = table_rect.height
        graph_h = max(240, int(right_h * 0.62))
        info_h = max(140, right_h - graph_h - 12)
        graph_rect = pg.Rect(right_x, top_y, right_w, graph_h)
        self._draw_graph(graph_rect, rows)

        info_rect = pg.Rect(right_x, graph_rect.bottom + 12, right_w, info_h)
        pg.draw.rect(self.screen, (20, 22, 27), info_rect)
        pg.draw.rect(self.screen, (70, 74, 86), info_rect, 1)
        info_title = self.font.render(
            _fit_text(self.font, "Other Information", info_rect.width - 20),
            True,
            (220, 220, 220),
        )
        self.screen.blit(info_title, (info_rect.x + 10, info_rect.y + 10))
        selected_row = self._selected_row()
        selected_step = "-"
        selected_status = "-"
        selected_master = "-"
        selected_env = "-"
        can_reopen = False
        can_close = False
        if isinstance(selected_row, dict):
            selected_step = str(int(selected_row.get("step_index", 0)) + 1)
            selected_status = str(selected_row.get("status", "pending"))
            master_id = selected_row.get("master_run_num")
            if master_id is not None:
                selected_master = f"master_{master_id}"
            env_dir = selected_row.get("env_dir")
            if env_dir:
                selected_env = str(env_dir)
            can_reopen = (
                master_id is not None
                and bool(selected_row.get("env_dir"))
            )
            can_close = bool(selected_row.get("reopen_open"))
        info_lines = [
            f"Current rate: {state.get('current_rate', '--')}",
            f"Current planned master: {state.get('current_planned_master', '--')}",
            f"Last completed master: {state.get('last_master', '--')}",
            f"Selected row: {selected_step}  status: {selected_status}",
            f"Selected master: {selected_master}",
            f"Selected env: {selected_env}",
            f"Graph-ready rows: {state.get('graph_ready_rows', 0)} / {len(rows)}",
            f"Summary file: {state.get('summary_path', '')}",
            f"Fit file: {state.get('fit_path', '')}",
            f"Hub stats file: {state.get('hub_stats_path', '')}",
            "Species/Frames refresh live. Controls: Update+Plot(U) Reopen(R) Close(C) Viewer(V) Close Hub(X) FPS(F), Copy buttons on equations",
            "Graph: selecting a row shows that master graph; click dots to inspect",
        ]
        if self._copy_status_text and ((time.time() - float(self._copy_status_ts)) <= 3.0):
            info_lines.append(self._copy_status_text)
        dot = self._selected_graph_point if isinstance(self._selected_graph_point, dict) else None
        if dot is not None:
            dot_type = "actual" if bool(dot.get("actual")) else "projected"
            step_idx = dot.get("step_index")
            step_label = (
                str(int(step_idx) + 1)
                if isinstance(step_idx, int)
                else "--"
            )
            dot_mode = str(dot.get("graph_mode", "hub"))
            rate_val = dot.get("x")
            x_text = f"{float(rate_val):.4f}" if _is_number(rate_val) else "--"
            dot_lines = [
                f"Dot: step {step_label} ({dot_type})",
                (
                    f"dot evo speed: {x_text}"
                    if dot_mode == "master"
                    else f"dot env rate: {x_text}"
                ),
            ]
            dot_y = dot.get("y")
            if _is_number(dot_y):
                dot_lines.append(f"dot y: {float(dot_y):.4f}")
            master_val = dot.get("master_run_num")
            if master_val is not None:
                dot_lines.append(f"dot master: master_{master_val}")
            status_val = dot.get("status")
            if status_val:
                dot_lines.append(f"dot status: {status_val}")
            if _is_number(dot.get("total_species")):
                dot_lines.append(f"dot species total: {float(dot.get('total_species')):.1f}")
            if _is_number(dot.get("max_species")):
                dot_lines.append(f"dot species max run: {float(dot.get('max_species')):.1f}")
            if _is_number(dot.get("max_frames")):
                dot_lines.append(f"dot frames max run: {float(dot.get('max_frames')):.0f}")
            if _is_number(dot.get("duration_s")):
                dot_lines.append(f"dot duration: {_fmt_duration(float(dot.get('duration_s')))}")
            info_lines.extend(dot_lines)
        button_y = info_rect.bottom - 42
        info_text_x = info_rect.x + 10
        info_text_w = info_rect.width - 20
        info_text_top = info_rect.y + 42
        info_text_bottom = button_y - 8
        line_h = 22
        max_info_lines = max(0, (info_text_bottom - info_text_top) // line_h)
        lines_to_render = list(info_lines)
        if len(lines_to_render) > max_info_lines:
            if max_info_lines <= 0:
                lines_to_render = []
            elif max_info_lines == 1:
                lines_to_render = ["..."]
            else:
                lines_to_render = lines_to_render[: max_info_lines - 1] + ["..."]
        yy = info_text_top
        for line in lines_to_render:
            surf = self.small.render(
                _fit_text(self.small, str(line), info_text_w),
                True,
                (185, 185, 185),
            )
            self.screen.blit(surf, (info_text_x, yy))
            yy += line_h

        button_gap = 8
        button_w = max(56, (info_rect.width - 20 - (3 * button_gap)) // 4)
        button_y = info_rect.bottom - 42
        button_x0 = info_rect.x + 10
        self._update_button_rect = self.pg.Rect(
            button_x0,
            button_y,
            button_w,
            30,
        )
        self._reopen_button_rect = self.pg.Rect(
            self._update_button_rect.right + button_gap,
            button_y,
            button_w,
            30,
        )
        self._close_button_rect = self.pg.Rect(
            self._reopen_button_rect.right + button_gap,
            button_y,
            button_w,
            30,
        )
        self._viewer_button_rect = self.pg.Rect(
            self._close_button_rect.right + button_gap,
            button_y,
            button_w,
            30,
        )
        upd_bg = (60, 80, 118)
        upd_fg = (230, 230, 230)
        pg.draw.rect(self.screen, upd_bg, self._update_button_rect)
        pg.draw.rect(self.screen, (150, 150, 150), self._update_button_rect, 1)
        upd_text = self.small.render(
            _fit_text(self.small, "Update+Plot (U)", self._update_button_rect.width - 12),
            True,
            upd_fg,
        )
        self.screen.blit(
            upd_text,
            (
                self._update_button_rect.x
                + (self._update_button_rect.width - upd_text.get_width()) // 2,
                self._update_button_rect.y + 7,
            ),
        )
        btn_bg = (40, 90, 60) if can_reopen else (42, 42, 46)
        btn_fg = (230, 230, 230) if can_reopen else (150, 150, 150)
        pg.draw.rect(self.screen, btn_bg, self._reopen_button_rect)
        pg.draw.rect(self.screen, (150, 150, 150), self._reopen_button_rect, 1)
        btn_text = self.small.render(
            _fit_text(self.small, "Reopen (R)", self._reopen_button_rect.width - 12),
            True,
            btn_fg,
        )
        self.screen.blit(
            btn_text,
            (
                self._reopen_button_rect.x
                + (self._reopen_button_rect.width - btn_text.get_width()) // 2,
                self._reopen_button_rect.y + 7,
            ),
        )
        close_bg = (110, 60, 60) if can_close else (42, 42, 46)
        close_fg = (230, 230, 230) if can_close else (150, 150, 150)
        pg.draw.rect(self.screen, close_bg, self._close_button_rect)
        pg.draw.rect(self.screen, (150, 150, 150), self._close_button_rect, 1)
        close_text = self.small.render(
            _fit_text(self.small, "Close (C)", self._close_button_rect.width - 12),
            True,
            close_fg,
        )
        self.screen.blit(
            close_text,
            (
                self._close_button_rect.x
                + (self._close_button_rect.width - close_text.get_width()) // 2,
                self._close_button_rect.y + 7,
            ),
        )
        viewer_bg = (66, 72, 120)
        viewer_fg = (230, 230, 230)
        pg.draw.rect(self.screen, viewer_bg, self._viewer_button_rect)
        pg.draw.rect(self.screen, (150, 150, 150), self._viewer_button_rect, 1)
        viewer_text = self.small.render(
            _fit_text(self.small, "Viewer (V)", self._viewer_button_rect.width - 12),
            True,
            viewer_fg,
        )
        self.screen.blit(
            viewer_text,
            (
                self._viewer_button_rect.x
                + (self._viewer_button_rect.width - viewer_text.get_width()) // 2,
                self._viewer_button_rect.y + 7,
            ),
        )

        pg.display.flip()
        draw_fps = self.uncapped_fps if self._click_uncap_active(time.time()) else self.draw_fps
        self.clock.tick(draw_fps)


def main() -> None:
    args = _parse_args()
    defaults = _hub_defaults_from_settings()
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    hub_root = _hub_container_dir(results_root)
    hub_root.mkdir(parents=True, exist_ok=True)
    _master_container_dir(results_root).mkdir(parents=True, exist_ok=True)
    _normal_sim_container_dir(results_root).mkdir(parents=True, exist_ok=True)
    removed_aliases = _cleanup_root_hub_alias_symlinks(results_root)
    if removed_aliases > 0:
        print(
            f"[hub] removed {removed_aliases} legacy root alias symlink(s); "
            "hub directories are now canonical."
        )
    selector_choice = None
    should_show_selector = (args.continue_hub is None) and ((not args.no_screen) or args.hub_select)
    if should_show_selector:
        selector_choice = _select_hub_run_ui(results_root)
        if selector_choice is None:
            print("Hub selection canceled.")
            return

    continue_hub_idx = None
    if args.continue_hub is not None:
        try:
            continue_hub_idx = max(0, int(args.continue_hub))
        except Exception:
            raise SystemExit("Invalid --continue-hub value.")
    elif isinstance(selector_choice, dict) and selector_choice.get("mode") == "continue":
        try:
            continue_hub_idx = max(0, int(selector_choice.get("hub_idx")))
        except Exception:
            continue_hub_idx = None

    continuing = continue_hub_idx is not None
    existing_hub_meta = {}
    if continuing:
        hub_idx = int(continue_hub_idx)
        hub_dir = hub_root / f"hub_{hub_idx}"
        legacy_hub_dir = results_root / f"hub_{hub_idx}"
        if (not hub_dir.is_dir()) and legacy_hub_dir.is_dir():
            hub_dir = legacy_hub_dir
        if not hub_dir.is_dir():
            raise SystemExit(f"Hub not found: {hub_dir}")
        hub_meta_path = hub_dir / "hub_meta.json"
        if not hub_meta_path.exists():
            raise SystemExit(f"Missing hub_meta.json for continuation: {hub_meta_path}")
        try:
            payload = json.loads(hub_meta_path.read_text())
        except Exception:
            raise SystemExit(f"Failed to parse {hub_meta_path}")
        if not isinstance(payload, dict):
            raise SystemExit(f"Invalid hub meta format: {hub_meta_path}")
        existing_hub_meta = payload
        start_rate = float(existing_hub_meta.get("start_rate", defaults["start_rate"]))
        end_rate = float(existing_hub_meta.get("end_rate", defaults["end_rate"]))
        step = float(existing_hub_meta.get("step", defaults["step"]))
        species_threshold = int(
            max(
                0,
                int(
                    args.species_threshold
                    if args.species_threshold is not None
                    else existing_hub_meta.get("species_threshold", defaults["species_threshold"])
                ),
            )
        )
        max_masters = int(existing_hub_meta.get("max_masters", defaults["max_masters"]))
        raw_rates = existing_hub_meta.get("rates")
        rates = []
        if isinstance(raw_rates, list):
            for val in raw_rates:
                try:
                    rates.append(float(val))
                except Exception:
                    continue
        if not rates:
            rates = _rate_values(start_rate, end_rate, step)
            if max_masters > 0:
                rates = rates[:max_masters]
    else:
        start_rate = float(args.start_rate if args.start_rate is not None else defaults["start_rate"])
        end_rate = float(args.end_rate if args.end_rate is not None else defaults["end_rate"])
        step = float(args.step if args.step is not None else defaults["step"])
        species_threshold = int(
            max(0, int(args.species_threshold if args.species_threshold is not None else defaults["species_threshold"]))
        )
        max_masters = int(args.max_masters if args.max_masters is not None else defaults["max_masters"])
        rates = _rate_values(start_rate, end_rate, step)
        if max_masters > 0:
            rates = rates[:max_masters]
        hub_idx = _allocate_hub_index(results_root)
        hub_dir = hub_root / f"hub_{hub_idx}"
        hub_dir.mkdir(parents=True, exist_ok=True)

    if not rates:
        raise SystemExit("No rates to run.")

    hub_meta_path = hub_dir / "hub_meta.json"
    hub_summary_path = hub_dir / "hub_summary.csv"
    fit_csv_path = hub_dir / "hub_fit_equations.csv"
    hub_stats_path = hub_dir / "hub_stats.csv"
    all_points_csv_path = hub_dir / "hub_all_points.csv"
    hub_all_points_weighted_by_fitness: list[tuple[float, float, float]] = []
    settings_snapshot = load_settings()
    try:
        master_cursor = int(settings_snapshot.get("num_tries_master", 0))
    except Exception:
        master_cursor = 0
    existing_master_ids = _collect_existing_master_ids(results_root)
    default_root = Path("results")
    try:
        same_root = default_root.resolve() == results_root.resolve()
    except Exception:
        same_root = str(default_root) == str(results_root)
    if not same_root:
        existing_master_ids.update(_collect_existing_master_ids(default_root))
    planned_master_ids = []
    raw_planned = existing_hub_meta.get("planned_master_ids") if continuing else None
    if isinstance(raw_planned, list):
        for idx in range(len(rates)):
            val = raw_planned[idx] if idx < len(raw_planned) else None
            try:
                planned_id = int(val)
            except Exception:
                planned_id = None
            planned_master_ids.append(planned_id)
    else:
        planned_master_ids = [None] * len(rates)
    if continuing:
        steps = existing_hub_meta.get("steps")
        if isinstance(steps, list):
            for step_info in steps:
                if not isinstance(step_info, dict):
                    continue
                try:
                    step_idx = int(step_info.get("step_index"))
                except Exception:
                    continue
                if step_idx < 0 or step_idx >= len(rates):
                    continue
                try:
                    planned_id = int(step_info.get("planned_master_run_num"))
                except Exception:
                    continue
                if planned_master_ids[step_idx] is None:
                    planned_master_ids[step_idx] = planned_id
    used_master_ids = set(int(v) for v in existing_master_ids)
    used_master_ids.update(int(v) for v in planned_master_ids if isinstance(v, int))
    missing_count = sum(1 for v in planned_master_ids if v is None)
    if missing_count > 0:
        filled = _plan_master_ids(master_cursor, missing_count, used_master_ids)
        fill_i = 0
        for idx in range(len(planned_master_ids)):
            if planned_master_ids[idx] is None and fill_i < len(filled):
                planned_master_ids[idx] = int(filled[fill_i])
                fill_i += 1
    planned_master_ids = [
        (None if v is None else int(v))
        for v in planned_master_ids
    ]
    planned_master_span = _format_id_span([v for v in planned_master_ids if isinstance(v, int)])
    if not continuing:
        reserved_master_count = len([v for v in planned_master_ids if isinstance(v, int)])
        if args.count is not None:
            sims_per_master = max(0, int(args.count))
        else:
            try:
                sims_per_master = int(settings_snapshot.get("simulations", {}).get("count", 3))
            except Exception:
                sims_per_master = 3
            sims_per_master = max(0, sims_per_master)
        reserved_sim_count = int(reserved_master_count * sims_per_master)
        reserved_master_next, reserved_sim_next = _reserve_global_counters(
            settings_snapshot=settings_snapshot,
            reserved_master_count=reserved_master_count,
            reserved_sim_count=reserved_sim_count,
            planned_master_ids=[v for v in planned_master_ids if isinstance(v, int)],
        )
        print(
            f"[hub_{hub_idx}] reserved counters: num_tries_master += {reserved_master_count}, "
            f"num_tries += {reserved_sim_count} -> master={reserved_master_next}, sim={reserved_sim_next}"
        )
    if continuing:
        hub_meta = dict(existing_hub_meta)
        hub_meta["hub_index"] = int(hub_idx)
        hub_meta["start_rate"] = float(start_rate)
        hub_meta["end_rate"] = float(end_rate)
        hub_meta["step"] = float(step)
        hub_meta["species_threshold"] = int(species_threshold)
        hub_meta["max_masters"] = int(max_masters)
        hub_meta["rates"] = rates
        hub_meta["planned_master_ids"] = planned_master_ids
        hub_meta["planned_master_range"] = planned_master_span
        if not isinstance(hub_meta.get("steps"), list):
            hub_meta["steps"] = []
        if not _is_number(hub_meta.get("created_at")):
            hub_meta["created_at"] = time.time()
        hub_meta["status"] = "running"
        hub_meta["resumed_at"] = time.time()
        try:
            hub_meta["resume_count"] = int(hub_meta.get("resume_count", 0)) + 1
        except Exception:
            hub_meta["resume_count"] = 1
    else:
        hub_meta = {
            "hub_index": int(hub_idx),
            "created_at": time.time(),
            "start_rate": float(start_rate),
            "end_rate": float(end_rate),
            "step": float(step),
            "species_threshold": int(species_threshold),
            "max_masters": int(max_masters),
            "rates": rates,
            "planned_master_ids": planned_master_ids,
            "planned_master_range": planned_master_span,
            "steps": [],
            "status": "running",
        }
    # Collapse duplicate step entries by step_index (keep latest occurrence).
    if isinstance(hub_meta.get("steps"), list):
        steps_raw = hub_meta.get("steps", [])
        last_pos_by_idx: dict[int, int] = {}
        for pos, step_info in enumerate(steps_raw):
            if not isinstance(step_info, dict):
                continue
            raw_idx = step_info.get("step_index")
            if not _is_number(raw_idx):
                continue
            last_pos_by_idx[int(raw_idx)] = int(pos)
        if last_pos_by_idx:
            deduped_steps = []
            for pos, step_info in enumerate(steps_raw):
                if not isinstance(step_info, dict):
                    continue
                raw_idx = step_info.get("step_index")
                if not _is_number(raw_idx):
                    deduped_steps.append(step_info)
                    continue
                if last_pos_by_idx.get(int(raw_idx)) != int(pos):
                    continue
                deduped_steps.append(step_info)
            hub_meta["steps"] = deduped_steps
    _write_json_file(hub_meta_path, hub_meta)
    _ensure_csv_with_header(
        hub_summary_path,
        "enviorment change rate,planned master run,master run,apex evolution rate,fitness,max species,fit sigma left,fit sigma right,fit r2\n",
    )
    _ensure_csv_with_header(
        fit_csv_path,
        "enviorment change rate,master run,apex x,apex y,sigma left,sigma right,r2,equation\n",
    )

    repo_root = Path(__file__).resolve().parent
    master_script = repo_root / "master_simulations.py"
    hub_viewer_script = repo_root / "hub_viewer.py"
    interpreter = sys.executable
    close_signal_dir = hub_dir / ".hub_control"
    close_signal_dir.mkdir(parents=True, exist_ok=True)
    shutdown_signal_env = os.environ.get("HUB_RUNNER_SHUTDOWN_FILE", "").strip()
    if shutdown_signal_env:
        hub_shutdown_path = Path(shutdown_signal_env)
    else:
        hub_shutdown_path = close_signal_dir / "shutdown_hub.signal"
    try:
        hub_shutdown_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        hub_shutdown_path.unlink(missing_ok=True)
    except Exception:
        pass
    simrun_launch_flag = str(os.environ.get("HUB_RUNNER_FROM_SIMRUN", "")).strip().lower()
    master_headless_mode = simrun_launch_flag in {"1", "true", "yes", "on"}
    reopened_master_procs = {}
    reopened_master_close_paths = {}
    hub_viewer_proc = None
    current_step_proc = None
    current_step_close_path = None
    current_step_close_requested = False
    abort_requested = False
    fatal_returncode = None

    def _upsert_hub_step(step_info: dict) -> None:
        if not isinstance(step_info, dict):
            return
        steps = hub_meta.get("steps")
        if not isinstance(steps, list):
            steps = []
            hub_meta["steps"] = steps
        raw_idx = step_info.get("step_index")
        if _is_number(raw_idx):
            target_idx = int(raw_idx)
            for pos, existing in enumerate(steps):
                if not isinstance(existing, dict):
                    continue
                existing_idx = existing.get("step_index")
                if not _is_number(existing_idx):
                    continue
                if int(existing_idx) == target_idx:
                    steps[pos] = dict(step_info)
                    return
        steps.append(dict(step_info))

    def _persist_hub_state(force: bool = True, clear_running_row: bool = False) -> None:
        if clear_running_row:
            dash_state["running_row"] = None
        dash_state["status"] = str(hub_meta.get("status", dash_state.get("status", "")))
        _write_json_file(hub_meta_path, hub_meta)
        _refresh_reopened_processes()
        dashboard.update(dash_state, force=force)

    def _master_subprocess_env(close_path: Path | None = None) -> dict:
        env = os.environ.copy()
        if isinstance(close_path, Path):
            env["MASTER_CLOSE_REQUEST_FILE"] = str(close_path)
        if master_headless_mode:
            # simRun launches hub_runner for terminal-only progress, so keep master
            # pygame windows hidden while preserving master execution.
            env.setdefault("SDL_VIDEODRIVER", "dummy")
            env.setdefault("SDL_AUDIODRIVER", "dummy")
        return env

    def _step_row_for_master(master_run_num: int):
        for item in step_rows:
            if item.get("master_run_num") == master_run_num:
                return item
        return None

    def _request_graceful_close(
        proc: subprocess.Popen, close_path: Path | None, label: str
    ) -> None:
        if proc is None:
            return
        if proc.poll() is not None:
            return
        if close_path is None:
            return
        try:
            close_path.write_text(str(time.time()))
            print(
                f"[hub_{hub_idx}] close requested for {label} (pid={proc.pid}) via {close_path}"
            )
        except Exception:
            pass

    def _close_all_reopened() -> None:
        for master_num, proc in list(reopened_master_procs.items()):
            close_path = reopened_master_close_paths.get(int(master_num))
            _request_graceful_close(proc, close_path, f"reopened master_{int(master_num)}")

    def _refresh_reopened_processes() -> None:
        stale = []
        for master_num, proc in reopened_master_procs.items():
            if proc.poll() is None:
                row = _step_row_for_master(master_num)
                if isinstance(row, dict):
                    row["reopen_open"] = True
                    row["reopen_pid"] = int(proc.pid)
            else:
                stale.append(master_num)
                row = _step_row_for_master(master_num)
                if isinstance(row, dict):
                    row["reopen_open"] = False
                    row["reopen_pid"] = None
        for key in stale:
            reopened_master_procs.pop(key, None)
            close_path = reopened_master_close_paths.pop(key, None)
            if isinstance(close_path, Path):
                try:
                    close_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def _reopen_master_row(row: dict) -> None:
        if not isinstance(row, dict):
            return
        master_run = row.get("master_run_num")
        env_dir = row.get("env_dir")
        if master_run is None or not env_dir:
            return
        cmd = [
            interpreter,
            str(master_script),
            "--results-dir",
            str(env_dir),
            "--continue-master-run",
            str(int(master_run)),
        ]
        master_dir = row.get("master_dir")
        if master_dir:
            cmd.extend(["--continue-master-dir", str(master_dir)])
        existing = reopened_master_procs.get(int(master_run))
        if existing is not None and existing.poll() is None:
            print(
                f"[hub_{hub_idx}] master_{int(master_run)} already open (pid={existing.pid})"
            )
            row["reopen_open"] = True
            row["reopen_pid"] = int(existing.pid)
            return
        print(
            f"[hub_{hub_idx}] reopen requested for master_{int(master_run)} @ {env_dir}"
        )
        close_path = close_signal_dir / f"close_master_{int(master_run)}.signal"
        try:
            close_path.unlink(missing_ok=True)
        except Exception:
            pass
        env = _master_subprocess_env(close_path)
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env)
        reopened_master_procs[int(master_run)] = proc
        reopened_master_close_paths[int(master_run)] = close_path
        row["reopen_open"] = True
        row["reopen_pid"] = int(proc.pid)

    def _close_master_row(row: dict) -> None:
        if not isinstance(row, dict):
            return
        master_run = row.get("master_run_num")
        if master_run is None:
            return
        proc = reopened_master_procs.get(int(master_run))
        close_path = reopened_master_close_paths.get(int(master_run))
        if proc is None:
            row["reopen_open"] = False
            row["reopen_pid"] = None
            return
        _request_graceful_close(proc, close_path, f"master_{int(master_run)}")

    def _open_hub_viewer() -> None:
        nonlocal hub_viewer_proc
        if hub_viewer_proc is not None and hub_viewer_proc.poll() is None:
            print(f"[hub_{hub_idx}] hub viewer already open (pid={hub_viewer_proc.pid})")
            return
        cmd = [
            interpreter,
            str(hub_viewer_script),
            "--results-root",
            str(results_root),
            "--hub-dir",
            str(hub_dir),
        ]
        try:
            hub_viewer_proc = subprocess.Popen(cmd, cwd=str(repo_root), env=os.environ.copy())
            print(f"[hub_{hub_idx}] opened hub viewer (pid={hub_viewer_proc.pid})")
        except Exception as exc:
            print(f"[hub_{hub_idx}] failed to open hub viewer: {exc}")

    def _terminate_process(proc: subprocess.Popen | None, label: str) -> None:
        if proc is None:
            return
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
        except Exception:
            return
        try:
            proc.wait(timeout=2.0)
            return
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass
        print(f"[hub_{hub_idx}] force-stopped {label} (pid={proc.pid})")

    def _cleanup_child_processes(close_viewer: bool = False) -> None:
        nonlocal hub_viewer_proc, current_step_proc, current_step_close_path, current_step_close_requested
        _close_all_reopened()
        if current_step_proc is not None and current_step_proc.poll() is None:
            _request_graceful_close(current_step_proc, current_step_close_path, "active hub step")
            current_step_close_requested = True

        deadline = time.time() + 2.0
        while time.time() < deadline:
            _refresh_reopened_processes()
            reopened_running = any(proc.poll() is None for proc in reopened_master_procs.values())
            step_running = current_step_proc is not None and current_step_proc.poll() is None
            if (not reopened_running) and (not step_running):
                break
            time.sleep(0.1)

        for master_num, proc in list(reopened_master_procs.items()):
            if proc.poll() is None:
                _terminate_process(proc, f"reopened master_{int(master_num)}")
        _refresh_reopened_processes()

        if current_step_proc is not None and current_step_proc.poll() is None:
            _terminate_process(current_step_proc, "active hub step")
        if current_step_proc is not None and current_step_proc.poll() is not None:
            current_step_proc = None
            if isinstance(current_step_close_path, Path):
                try:
                    current_step_close_path.unlink(missing_ok=True)
                except Exception:
                    pass
            current_step_close_path = None
            current_step_close_requested = False

        if close_viewer:
            if hub_viewer_proc is not None and hub_viewer_proc.poll() is None:
                _terminate_process(hub_viewer_proc, "hub viewer")
            if hub_viewer_proc is not None and hub_viewer_proc.poll() is not None:
                hub_viewer_proc = None

    def _apex_from_points(points: list[tuple[float, float]]) -> tuple[float | None, float | None]:
        if not points:
            return (None, None)
        try:
            x_val, y_val = max(points, key=lambda p: p[1])
        except Exception:
            return (None, None)
        if not (_is_number(x_val) and _is_number(y_val)):
            return (None, None)
        return (float(x_val), float(y_val))

    def _refresh_row_runtime_from_disk(row: dict):
        if not isinstance(row, dict):
            return (None, None, [])
        env_dir_raw = row.get("env_dir")
        if not env_dir_raw:
            return (None, None, [])
        env_dir = Path(str(env_dir_raw))
        if not env_dir.is_dir():
            return (None, None, [])
        preferred_runs: list[int | None] = []
        raw_master_run = row.get("master_run_num")
        raw_planned_run = row.get("planned_master_run_num")
        preferred_runs.append(int(raw_master_run) if _is_number(raw_master_run) else None)
        preferred_runs.append(int(raw_planned_run) if _is_number(raw_planned_run) else None)
        master_dir = _select_master_dir(env_dir, preferred_runs)
        run_nums = _master_run_nums(master_dir) if master_dir is not None else []
        if not run_nums:
            run_nums = _discover_env_run_nums(env_dir)
        if not run_nums and master_dir is None:
            return (None, None, [])
        max_species = _max_species(env_dir, run_nums)
        total_species = _total_species(env_dir, run_nums)
        max_frames = _max_frames(env_dir, run_nums)
        max_elapsed = _max_elapsed_seconds(env_dir, run_nums)
        if master_dir is not None:
            try:
                master_run_num = int(master_dir.name.split("_", 1)[1])
            except Exception:
                master_run_num = None
            row["master_dir"] = str(master_dir)
        else:
            master_run_num = None
        if master_run_num is not None:
            row["master_run_num"] = master_run_num
        if row.get("planned_master_run_num") is None and master_run_num is not None:
            row["planned_master_run_num"] = master_run_num
        row["max_species"] = max_species
        row["total_species"] = total_species
        row["max_frames"] = max_frames
        row["run_nums"] = run_nums
        if _is_number(max_elapsed):
            row["duration_s"] = float(max_elapsed)
            if not (str(row.get("status", "")) == "running" and _is_number(row.get("started_at"))):
                row["duration_base_s"] = float(max_elapsed)
        return (env_dir, master_dir, run_nums)

    def _refresh_row_from_disk(row: dict, rebuild_master_combined: bool = False) -> None:
        env_dir, master_dir, run_nums = _refresh_row_runtime_from_disk(row)
        if env_dir is None:
            return
        if master_dir is not None:
            if rebuild_master_combined:
                try:
                    _rebuild_master_combined_for_update(env_dir, master_dir, run_nums)
                except Exception:
                    pass
            points = _master_points(master_dir, run_nums)
        else:
            points = []
            for run_num in run_nums:
                points.extend(
                    _extract_points_from_csv(
                        env_dir / str(run_num) / f"parsedArithmeticMeanSimulatino{run_num}_Log.csv"
                    )
                )
        if not points:
            points = _snapshot_points_from_runs(env_dir, run_nums)
        fit = _fit_stitched_gaussian(points)
        if fit is None:
            apex_x, apex_y = _apex_from_points(points)
        else:
            apex_x = fit.get("apex_x")
            apex_y = fit.get("apex_y")
        row["apex_evolution_rate"] = apex_x if _is_number(apex_x) else None
        row["apex_fitness"] = apex_y if _is_number(apex_y) else None
        row["fit"] = fit
        row["points"] = points
        row["point_count"] = int(len(points))

    def _manual_update_rows_from_disk(
        rebuild_master_combined: bool = False,
        selected_row: dict | None = None,
        export_graphs: bool = False,
    ) -> None:
        nonlocal hub_all_points_weighted_by_fitness
        ready_rows = 0
        for row in step_rows:
            _refresh_row_from_disk(row, rebuild_master_combined=rebuild_master_combined)
            if _is_number(row.get("apex_evolution_rate")) and _is_number(row.get("apex_fitness")):
                ready_rows += 1
        hub_all_points_weighted_by_fitness = _hub_all_points_weighted_by_fitness(step_rows)
        _write_hub_fit_equations_csv(fit_csv_path, step_rows)
        _write_hub_all_points_csv(all_points_csv_path, step_rows)
        _write_hub_stats_csv(hub_stats_path, step_rows)
        if _sync_hub_meta_steps_from_rows(hub_meta, step_rows):
            try:
                _write_json_file(hub_meta_path, hub_meta)
            except Exception:
                pass
        rows_with_master = [
            row for row in step_rows if isinstance(row, dict) and row.get("master_run_num") is not None
        ]
        if rows_with_master:
            last_row = max(rows_with_master, key=lambda r: int(r.get("step_index", -1)))
            try:
                dash_state["last_master"] = f"master_{int(last_row.get('master_run_num'))}"
            except Exception:
                pass
        dash_state["graph_ready_rows"] = int(ready_rows)
        if rebuild_master_combined:
            print(
                f"[hub_{hub_idx}] update rebuilt combined master logs/means and refreshed graph data "
                f"({ready_rows}/{len(step_rows)} rows with graph points)"
            )
        else:
            print(
                f"[hub_{hub_idx}] manual update refreshed hub dashboard data "
                f"({ready_rows}/{len(step_rows)} rows with graph points)"
            )
        if not export_graphs:
            return

        selected_ref = None
        if isinstance(selected_row, dict):
            selected_step = selected_row.get("step_index")
            if _is_number(selected_step):
                for candidate in step_rows:
                    if int(candidate.get("step_index", -1)) == int(selected_step):
                        selected_ref = candidate
                        break
            if selected_ref is None and selected_row in step_rows:
                selected_ref = selected_row

        if isinstance(selected_ref, dict):
            master_num = selected_ref.get("master_run_num")
            step_idx = selected_ref.get("step_index")
            if _is_number(master_num):
                selected_name = f"u_master_{int(master_num)}_graph.png"
            elif _is_number(step_idx):
                selected_name = f"u_step_{int(step_idx) + 1:03d}_master_graph.png"
            else:
                selected_name = "u_selected_master_graph.png"
            selected_path = hub_dir / selected_name
            if _plot_selected_master_view(selected_ref, selected_path):
                print(f"[hub_{hub_idx}] wrote selected master graph: {selected_path}")
                if _open_path_with_default_app(selected_path):
                    print(f"[hub_{hub_idx}] opened selected master graph window")
            else:
                print(
                    f"[hub_{hub_idx}] selected row has no graphable points yet; "
                    "skipped selected master graph export"
                )
        else:
            print(f"[hub_{hub_idx}] no selected row; skipped selected master graph export")

        full_hub_path = hub_dir / "u_full_hub_view.png"
        if _plot_hub_full_view(step_rows, full_hub_path):
            print(f"[hub_{hub_idx}] wrote full hub graph: {full_hub_path}")
            if _open_path_with_default_app(full_hub_path):
                print(f"[hub_{hub_idx}] opened full hub graph window")
        else:
            print(f"[hub_{hub_idx}] full hub graph export skipped (not enough data)")

        # Keep a persistent hub-level graph UI open for inspection after U.
        _open_hub_viewer()

    def _shutdown_hub_run() -> None:
        nonlocal abort_requested, current_step_proc, current_step_close_requested
        if abort_requested:
            return
        abort_requested = True
        print(f"[hub_{hub_idx}] close hub requested")
        _close_all_reopened()
        if current_step_proc is not None:
            _request_graceful_close(
                current_step_proc,
                current_step_close_path,
                "active hub step",
            )
            current_step_close_requested = True

    def _consume_external_shutdown_signal() -> bool:
        try:
            if not hub_shutdown_path.exists():
                return False
        except Exception:
            return False
        try:
            hub_shutdown_path.unlink(missing_ok=True)
        except Exception:
            pass
        return True

    hub_rows = []
    step_rows = []
    step_meta_by_index = {}
    if continuing:
        raw_steps = existing_hub_meta.get("steps")
        if isinstance(raw_steps, list):
            for step_info in raw_steps:
                if not isinstance(step_info, dict):
                    continue
                raw_idx = step_info.get("step_index")
                if not _is_number(raw_idx):
                    continue
                step_meta_by_index[int(raw_idx)] = step_info
    for idx, rate in enumerate(rates):
        env_dir = hub_dir / _rate_label(rate)
        row = {
            "step_index": int(idx),
            "env_rate": float(rate),
            "planned_master_run_num": (
                planned_master_ids[idx] if idx < len(planned_master_ids) else None
            ),
            "master_run_num": None,
            "status": "pending",
            "max_species": None,
            "total_species": None,
            "max_frames": None,
            "apex_fitness": None,
            "apex_evolution_rate": None,
            "started_at": None,
            "finished_at": None,
            "duration_s": None,
            "duration_base_s": 0.0,
            "env_dir": str(env_dir),
            "master_dir": None,
            "resume_existing": False,
            "reopen_open": False,
            "reopen_pid": None,
        }
        if continuing and env_dir.is_dir():
            _refresh_row_from_disk(row)
            has_existing_master = (
                row.get("master_run_num") is not None and bool(row.get("master_dir"))
            )
            step_meta = step_meta_by_index.get(int(idx))
            recorded_status = (
                str(step_meta.get("status", "")).strip().lower()
                if isinstance(step_meta, dict)
                else ""
            )
            recorded_master_run = (
                step_meta.get("master_run_num")
                if isinstance(step_meta, dict)
                else None
            )
            master_matches_record = True
            if recorded_status == "ok" and _is_number(recorded_master_run):
                if row.get("master_run_num") is None:
                    master_matches_record = False
                else:
                    master_matches_record = int(row.get("master_run_num")) == int(recorded_master_run)
            step_complete = bool(
                recorded_status == "ok"
                and has_existing_master
                and master_matches_record
            )
            if step_complete:
                row["status"] = "ok"
                row["resume_existing"] = False
                hub_rows.append(
                    {
                        "env_rate": float(rate),
                        "apex_x": row.get("apex_evolution_rate"),
                        "apex_y": row.get("apex_fitness"),
                        "fit": row.get("fit"),
                        "points": row.get("points") or [],
                    }
                )
            else:
                row["status"] = "pending"
                row["resume_existing"] = bool(has_existing_master)
        step_rows.append(row)
    dashboard = _HubDashboard(
        enabled=(not args.no_screen),
        hub_idx=hub_idx,
        hub_dir=hub_dir,
        planned_master_span=planned_master_span,
        species_threshold=species_threshold,
        update_callback=lambda row=None: _manual_update_rows_from_disk(
            rebuild_master_combined=True,
            selected_row=row,
            export_graphs=True,
        ),
        reopen_callback=_reopen_master_row,
        close_callback=_close_master_row,
        viewer_callback=_open_hub_viewer,
        shutdown_callback=_shutdown_hub_run,
    )
    dash_state = {
        "start_ts": float(hub_meta["created_at"]),
        "rows": step_rows,
        "status": hub_meta["status"],
        "running_row": None,
        "current_rate": "--",
        "current_planned_master": "--",
        "last_master": "--",
        "summary_path": str(hub_summary_path),
        "fit_path": str(fit_csv_path),
        "hub_stats_path": str(hub_stats_path),
        "graph_ready_rows": 0,
    }
    if continuing:
        print(f"[hub_{hub_idx}] continuing hub run at {hub_dir}")
    else:
        print(f"[hub_{hub_idx}] creating hub run at {hub_dir}")
    print(
        f"[hub_{hub_idx}] planned master ids for this hub: {planned_master_span}"
    )
    completed_rows = [row for row in step_rows if str(row.get("status", "")) == "ok"]
    pending_rows = [row for row in step_rows if str(row.get("status", "")) != "ok"]
    if completed_rows:
        last_done = sorted(
            completed_rows,
            key=lambda row: int(row.get("step_index", -1)),
        )[-1]
        master_num = last_done.get("master_run_num")
        if master_num is not None:
            dash_state["last_master"] = f"master_{int(master_num)}"
    if continuing:
        print(
            f"[hub_{hub_idx}] continuation progress: {len(completed_rows)}/{len(step_rows)} "
            f"completed, {len(pending_rows)} pending"
        )
    _refresh_reopened_processes()
    _manual_update_rows_from_disk()
    dashboard.update(dash_state, force=True)

    for step_idx, rate in enumerate(rates):
        if _consume_external_shutdown_signal():
            _shutdown_hub_run()
        if abort_requested:
            break
        row_ref = step_rows[step_idx]
        if str(row_ref.get("status", "")) == "ok":
            continue
        row_ref["status"] = "running"
        row_ref["started_at"] = time.time()
        base_dur = row_ref.get("duration_s")
        row_ref["duration_base_s"] = float(base_dur) if _is_number(base_dur) else 0.0
        dash_state["status"] = "running"
        dash_state["running_row"] = int(step_idx) + 1
        dash_state["current_rate"] = f"{float(rate):.2f}"
        env_dir = hub_dir / _rate_label(rate)
        env_dir.mkdir(parents=True, exist_ok=True)
        row_ref["env_dir"] = str(env_dir)
        planned_master_run = (
            planned_master_ids[step_idx] if step_idx < len(planned_master_ids) else None
        )
        row_ref["planned_master_run_num"] = planned_master_run
        dash_state["current_planned_master"] = (
            "--" if planned_master_run is None else str(int(planned_master_run))
        )
        resume_existing = bool(row_ref.get("resume_existing"))
        existing_master_run = row_ref.get("master_run_num")
        existing_master_dir = row_ref.get("master_dir")
        if resume_existing and existing_master_run is not None:
            cmd = [
                interpreter,
                str(master_script),
                "--non-interactive",
                "--results-dir",
                str(env_dir),
                "--continue-master-run",
                str(int(existing_master_run)),
                "--env-change-rate",
                str(rate),
                "--species-stop",
                str(species_threshold),
                "--script",
                str(args.script),
            ]
            if existing_master_dir:
                cmd.extend(["--continue-master-dir", str(existing_master_dir)])
        else:
            cmd = [
                interpreter,
                str(master_script),
                "--non-interactive",
                "--results-dir",
                str(env_dir),
                "--env-change-rate",
                str(rate),
                "--species-stop",
                str(species_threshold),
                "--script",
                str(args.script),
            ]
            if planned_master_run is not None:
                cmd.extend(["--master-run-num", str(int(planned_master_run))])
            if args.count is not None:
                cmd.extend(["--count", str(int(args.count))])

        if resume_existing and existing_master_run is not None:
            print(
                f"[hub_{hub_idx}] step {step_idx + 1}/{len(rates)} "
                f"rate={rate:.2f} continuing master_{int(existing_master_run)}"
            )
        else:
            print(
                f"[hub_{hub_idx}] step {step_idx + 1}/{len(rates)} "
                f"rate={rate:.2f} planned_master={'' if planned_master_run is None else planned_master_run}"
            )
        current_step_close_path = close_signal_dir / f"close_step_{int(step_idx)}.signal"
        try:
            current_step_close_path.unlink(missing_ok=True)
        except Exception:
            pass
        current_step_close_requested = False
        env = _master_subprocess_env(current_step_close_path)
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env)
        current_step_proc = proc
        live_metrics_refresh_s = 0.5
        last_live_metrics_refresh = 0.0
        while proc.poll() is None:
            if _consume_external_shutdown_signal():
                _shutdown_hub_run()
            if abort_requested and (not current_step_close_requested):
                _request_graceful_close(
                    proc,
                    current_step_close_path,
                    f"hub step {step_idx + 1}",
                )
                current_step_close_requested = True
            now_ts = time.time()
            if (now_ts - last_live_metrics_refresh) >= live_metrics_refresh_s:
                _refresh_row_runtime_from_disk(row_ref)
                for candidate in step_rows:
                    if candidate is row_ref:
                        continue
                    if bool(candidate.get("reopen_open")):
                        _refresh_row_runtime_from_disk(candidate)
                last_live_metrics_refresh = now_ts
            _refresh_reopened_processes()
            dashboard.update(dash_state)
            sleep_s = dashboard.poll_sleep_seconds() if dashboard.enabled else 0.12
            if sleep_s > 0:
                time.sleep(sleep_s)
        returncode = int(proc.returncode if proc.returncode is not None else 1)
        current_step_proc = None
        if isinstance(current_step_close_path, Path):
            try:
                current_step_close_path.unlink(missing_ok=True)
            except Exception:
                pass
        current_step_close_path = None
        current_step_close_requested = False
        row_ref["finished_at"] = time.time()
        if _is_number(row_ref.get("started_at")):
            row_ref["duration_s"] = max(
                0.0, float(row_ref["finished_at"]) - float(row_ref["started_at"])
            )
        base_dur = row_ref.get("duration_base_s")
        if _is_number(base_dur) and _is_number(row_ref.get("duration_s")):
            row_ref["duration_s"] = float(base_dur) + float(row_ref["duration_s"])

        step_info = {
            "step_index": int(step_idx),
            "env_rate": float(rate),
            "env_dir": str(env_dir),
            "planned_master_run_num": planned_master_run,
            "returncode": returncode,
            "finished_at": time.time(),
        }
        if abort_requested:
            step_info["status"] = "aborted"
            row_ref["status"] = "aborted"
            _upsert_hub_step(step_info)
            hub_meta["status"] = "aborted_by_user"
            hub_meta["aborted_at"] = time.time()
            _persist_hub_state(force=True, clear_running_row=True)
            break
        if returncode != 0:
            step_info["status"] = "failed"
            row_ref["status"] = "failed"
            _upsert_hub_step(step_info)
            hub_meta["status"] = "failed"
            _persist_hub_state(force=True, clear_running_row=True)
            fatal_returncode = int(returncode)
            break

        selected_runs: list[int | None] = []
        selected_runs.append(int(planned_master_run) if _is_number(planned_master_run) else None)
        selected_runs.append(
            int(row_ref.get("master_run_num")) if _is_number(row_ref.get("master_run_num")) else None
        )
        master_dir = _select_master_dir(env_dir, selected_runs)
        if master_dir is None:
            step_info["status"] = "no_master"
            row_ref["status"] = "no_master"
            _upsert_hub_step(step_info)
            hub_meta["status"] = "stopped_no_master"
            _persist_hub_state(force=True, clear_running_row=True)
            break

        run_nums = _master_run_nums(master_dir)
        max_species = _max_species(env_dir, run_nums)
        total_species = _total_species(env_dir, run_nums)
        max_frames = _max_frames(env_dir, run_nums)
        points = _master_points(master_dir, run_nums)
        fit = _fit_stitched_gaussian(points)
        if fit is None:
            apex_x, apex_y = (None, None)
        else:
            apex_x = fit["apex_x"]
            apex_y = fit["apex_y"]
        try:
            master_run_num = int(master_dir.name.split("_", 1)[1])
        except Exception:
            master_run_num = None

        with open(hub_summary_path, "a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    rate,
                    planned_master_run if planned_master_run is not None else "",
                    master_run_num if master_run_num is not None else "",
                    apex_x if apex_x is not None else "",
                    apex_y if apex_y is not None else "",
                    max_species if max_species is not None else "",
                    (fit["sigma_left"] if fit else ""),
                    (fit["sigma_right"] if fit else ""),
                    (fit["r2"] if fit else ""),
                ]
            )

        if fit:
            with open(fit_csv_path, "a", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        rate,
                        master_run_num if master_run_num is not None else "",
                        fit["apex_x"],
                        fit["apex_y"],
                        fit["sigma_left"],
                        fit["sigma_right"],
                        fit["r2"] if fit["r2"] is not None else "",
                        fit["equation"],
                    ]
                )

        step_info.update(
            {
                "status": "ok",
                "master_dir": str(master_dir),
                "master_run_num": master_run_num,
                "planned_master_run_num": planned_master_run,
                "run_nums": run_nums,
                "max_species": max_species,
                "total_species": total_species,
                "max_frames": max_frames,
                "apex_evolution_rate": apex_x,
                "apex_fitness": apex_y,
                "fit": fit,
                "point_count": len(points),
            }
        )
        row_ref["status"] = "ok"
        row_ref["resume_existing"] = False
        row_ref["master_run_num"] = master_run_num
        row_ref["master_dir"] = str(master_dir)
        row_ref["max_species"] = max_species
        row_ref["total_species"] = total_species
        row_ref["max_frames"] = max_frames
        row_ref["apex_evolution_rate"] = apex_x
        row_ref["apex_fitness"] = apex_y
        row_ref["fit"] = fit
        row_ref["points"] = points
        row_ref["point_count"] = int(len(points))
        dash_state["last_master"] = (
            "--" if master_run_num is None else f"master_{int(master_run_num)}"
        )
        dash_state["running_row"] = None
        if (
            planned_master_run is not None
            and master_run_num is not None
            and int(planned_master_run) != int(master_run_num)
        ):
            step_info["master_run_mismatch"] = True
            step_info["master_run_mismatch_note"] = (
                f"expected master_{planned_master_run}, got master_{master_run_num}"
            )
        _upsert_hub_step(step_info)
        _write_json_file(hub_meta_path, hub_meta)

        hub_rows.append(
            {
                "env_rate": float(rate),
                "apex_x": apex_x,
                "apex_y": apex_y,
                "fit": fit,
                "points": points,
            }
        )
        hub_all_points_weighted_by_fitness = _hub_all_points_weighted_by_fitness(step_rows)
        _write_hub_all_points_csv(all_points_csv_path, step_rows)
        _write_hub_stats_csv(hub_stats_path, step_rows)
        _refresh_reopened_processes()
        dashboard.update(dash_state, force=True)
    else:
        hub_meta["status"] = "completed"
        hub_meta["completed_at"] = time.time()
        _persist_hub_state(force=True, clear_running_row=False)

    if abort_requested and str(hub_meta.get("status", "")) == "running":
        hub_meta["status"] = "aborted_by_user"
        hub_meta["aborted_at"] = time.time()
        _persist_hub_state(force=True, clear_running_row=True)

    if (not args.skip_plots) and (not abort_requested) and (fatal_returncode is None):
        plotted = []
        scatter_path = hub_dir / "hub_apex_scatter.png"
        if _plot_hub_scatter(hub_rows, scatter_path):
            plotted.append(scatter_path.name)
        fits_path = hub_dir / "hub_stitched_fits.png"
        if _plot_stitched_fits(hub_rows, fits_path):
            plotted.append(fits_path.name)
        ratio_path = hub_dir / "hub_ratio_curve.png"
        if _plot_ratio_curve(hub_rows, ratio_path):
            plotted.append(ratio_path.name)
        if plotted:
            hub_meta["plots"] = plotted
            _persist_hub_state(force=True, clear_running_row=False)

    # When a hub finishes all steps, keep the dashboard open by default
    # so users can inspect the final graph and reopen masters if needed.
    hold_seconds = float(args.screen_hold_seconds)
    if dashboard.enabled and (not abort_requested) and str(hub_meta.get("status", "")) == "completed":
        if hold_seconds < 0.0:
            while dashboard.enabled:
                _refresh_reopened_processes()
                dashboard.update(dash_state, force=True)
                time.sleep(0.1)
        elif hold_seconds > 0.0:
            hold_until = time.time() + hold_seconds
            while dashboard.enabled and time.time() < hold_until:
                _refresh_reopened_processes()
                dashboard.update(dash_state, force=True)
                time.sleep(0.1)

    final_status = str(hub_meta.get("status", ""))
    if fatal_returncode is not None or final_status in {"failed", "aborted_by_user", "stopped_no_master"}:
        _cleanup_child_processes(close_viewer=True)

    if fatal_returncode is not None:
        print(f"Hub failed (returncode={int(fatal_returncode)}): {hub_dir}")
    elif abort_requested:
        print(f"Hub aborted by user: {hub_dir}")
    elif final_status == "stopped_no_master":
        print(f"Hub stopped (no master produced): {hub_dir}")
    else:
        print(f"Hub complete: {hub_dir}")
    print(f"Summary: {hub_summary_path}")
    print(f"Equations: {fit_csv_path}")
    print(f"Hub stats: {hub_stats_path}")
    print(f"All datapoints: {all_points_csv_path}")
    print(
        "Weighted datapoints kept in memory variable "
        f"`hub_all_points_weighted_by_fitness` ({len(hub_all_points_weighted_by_fitness)} rows)"
    )
    if fatal_returncode is not None:
        raise SystemExit(int(fatal_returncode))


if __name__ == "__main__":
    main()
