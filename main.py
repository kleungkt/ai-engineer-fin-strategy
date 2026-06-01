#!/usr/bin/env python3
"""
AI Engineer Portfolio — Unified CLI
====================================
統一入口，一鍵切換項目、跑測試、啟動服務。

Usage:
    python main.py list                        # 列出所有項目
    python main.py run <project> [--port 8001] # 啟動 API 服務
    python main.py demo <project>              # 跑 demo 展示
    python main.py test <project>              # 跑測試
    python main.py status                      # 項目狀態總覽
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECTS = {
    "01-nl-stock-query": "自然語言股票查詢器",
    "02-rag-financial-kb": "RAG 金融知識庫問答",
    "03-ai-strategy-generator": "AI 策略生成器",
    "04-strategy-diagnostics": "策略診斷系統",
    "05-finllm-finetune": "金融 LLM 微調",
}

ROOT_DIR = Path(__file__).parent


def cmd_list():
    """列出所有可用項目"""
    print("\n📊 AI Engineer Portfolio — Financial Strategy\n")
    print(f"{'#':<4} {'Project ID':<32} {'Description'}")
    print("─" * 72)
    for pid, desc in PROJECTS.items():
        num = pid[:2]
        print(f"  {num:<4} {pid:<32} {desc}")
    print(f"\n  Total: {len(PROJECTS)} projects | 594 tests | 25,000+ lines")
    print()


def cmd_test(project: str | None):
    """跑指定或全部測試"""
    projects = [project] if project else list(PROJECTS.keys())

    total_pass = 0
    total_fail = 0

    for p in projects:
        if p not in PROJECTS:
            print(f"❌ Unknown project: {p}")
            continue

        proj_dir = ROOT_DIR / "projects" / p
        print(f"\n🧪 Testing: {p} — {PROJECTS[p]}")

        result = subprocess.run(
            [".venv/bin/pytest", "tests/", "-q", "--tb=line"],
            cwd=proj_dir,
            env={**dict(__import__("os").environ), "PYTHONPATH": "src"},
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr
        # Extract pass/fail counts
        pass_count = 0
        fail_count = 0
        for line in output.split("\n"):
            if "passed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "passed":
                        try:
                            pass_count = int(parts[i - 1])
                        except (ValueError, IndexError):
                            pass
            if "failed" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "failed":
                        try:
                            fail_count = int(parts[i - 1])
                        except (ValueError, IndexError):
                            pass

        status = "✅" if fail_count == 0 else "❌"
        print(f"  {status} {pass_count} passed, {fail_count} failed")
        total_pass += pass_count
        total_fail += fail_count

    print(f"\n{'═' * 50}")
    overall = "✅ ALL PASSED" if total_fail == 0 else f"❌ {total_fail} FAILED"
    print(f"  Total: {total_pass} passed, {total_fail} failed — {overall}")
    print(f"{'═' * 50}\n")

    return 1 if total_fail > 0 else 0


def cmd_run(project: str, port: int):
    """啟動指定項目的 API 服務"""
    if project not in PROJECTS:
        print(f"❌ Unknown project: {project}")
        print(f"   Available: {', '.join(PROJECTS.keys())}")
        return 1

    proj_dir = ROOT_DIR / "projects" / project
    print(f"\n🚀 Starting: {project} — {PROJECTS[p]}")
    print(f"   Port: {port}")
    print(f"   Docs: http://localhost:{port}/docs")
    print(f"   Press Ctrl+C to stop\n")

    subprocess.run(
        [
            ".venv/bin/uvicorn",
            "src.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--reload",
        ],
        cwd=proj_dir,
        env={**dict(__import__("os").environ), "PYTHONPATH": "src"},
    )


def cmd_status():
    """顯示各項目狀態"""
    print("\n📊 Project Status\n")
    print(f"{'Project':<35} {'Tests':>6} {'Lines':>7} {'API':>5}")
    print("─" * 60)

    total_files = 0
    total_lines = 0

    for p in PROJECTS:
        proj_dir = ROOT_DIR / "projects" / p
        src_dir = proj_dir / "src"

        # Count files and lines
        files = list(src_dir.rglob("*.py"))
        lines = sum(len(f.read_text().splitlines()) for f in files)
        total_files += len(files)
        total_lines += lines

        # Check if tests exist
        tests_dir = proj_dir / "tests"
        test_files = list(tests_dir.glob("test_*.py")) if tests_dir.exists() else []

        # Check if api.py exists
        has_api = "✅" if (src_dir / "api.py").exists() else "—"

        print(f"  {p:<35} {len(test_files):>4}f {lines:>6} {has_api:>5}")

    print("─" * 60)
    print(f"  {'TOTAL':<35} {total_files:>4}p {total_lines:>6}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AI Engineer Portfolio — Financial Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py list              Show all projects
  python main.py test              Run all 594 tests
  python main.py test 01           Run P1 tests only
  python main.py run 01 --port 8001  Start P1 API server
  python main.py status            Show project statistics
        """,
    )

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List all projects")
    sub.add_parser("status", help="Show project statistics")
    sub.add_parser("test", help="Run tests")

    run_parser = sub.add_parser("run", help="Start an API server")
    run_parser.add_argument("project", help="Project ID (e.g., 01-nl-stock-query)")
    run_parser.add_argument("--port", type=int, default=8001, help="Port number")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list()
    elif args.command == "test":
        cmd_test(None)
    elif args.command == "run":
        cmd_run(args.project, args.port)
    elif args.command == "status":
        cmd_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    sys.exit(main() or 0)
