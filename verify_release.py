"""Проверка кодовой и документальной упаковки проекта без фиктивных результатов."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATIC_REQUIRED = [
    "README_RESEARCH.md", "DISSERTATION.md", "DISSERTATION_PROJECT_PLAN.md",
    "DATA_DICTIONARY.md", "REFERENCES.md", "research_protocol.md",
    "requirements-research.txt", "eda.py", "run_experiments.py", "robustness.py",
    "interpret_results.py", "verify_release.py", "tests/test_pipeline.py", "tests/test_eda.py",
    "tests/test_robustness.py", "tests/test_interpret_results.py",
]


def _run(command: list[str]) -> tuple[bool, str]:
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return completed.returncode == 0, output[-2500:]


def verify(strict: bool = False) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    missing = [name for name in STATIC_REQUIRED if not (ROOT / name).exists()]
    checks.append({"name": "static_required_files", "ok": not missing, "missing": missing})

    encoding_errors = []
    for name in STATIC_REQUIRED:
        path = ROOT / name
        if not path.exists():
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            encoding_errors.append({"file": name, "error": str(exc)})
    checks.append({"name": "utf8_source_and_docs", "ok": not encoding_errors, "errors": encoding_errors})

    compile_ok, compile_output = _run([sys.executable, "-m", "compileall", "-q", "."])
    checks.append({"name": "compileall", "ok": compile_ok, "output": compile_output})
    tests_ok, tests_output = _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    checks.append({"name": "unit_tests", "ok": tests_ok, "output": tests_output})

    panel_exists = (ROOT / "panel.csv").exists()
    official_names = ("model_metrics.json", "robustness_results.json",
                      "hypothesis_results.json", "hypothesis_results.csv")
    official_artifacts = [name for name in official_names if (ROOT / name).exists()]
    artifacts_consistent = panel_exists or not official_artifacts
    checks.append({"name": "no_official_results_without_panel", "ok": artifacts_consistent,
                   "panel_exists": panel_exists, "official_artifacts": official_artifacts})

    suspicious = []
    secret_names = {"cookies.json", "cookies.txt", ".env", "storage_state.json"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".venv" in path.parts or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.name.lower() in secret_names or "browser_profile" in path.parts:
            suspicious.append(str(path.relative_to(ROOT)))
    checks.append({"name": "no_local_secrets_or_browser_profiles", "ok": not suspicious,
                   "paths": suspicious})

    checks_ok = all(bool(item["ok"]) for item in checks)
    ready = checks_ok and panel_exists
    result = {
        "protocol_version": "release-check-1.0",
        "strict": strict,
        "status": "release_ready" if ready else "preflight_ok_no_real_panel" if checks_ok else "failed",
        "checks": checks,
        "note": "Проверки кода не заменяют полный запуск и ручной аудит реальной панели.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверить кодовую и документальную упаковку проекта")
    parser.add_argument("--strict", action="store_true", help="считать отсутствие реальной panel.csv блокирующей ошибкой")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    result = verify(strict=args.strict)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    if args.strict and result["status"] != "release_ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()