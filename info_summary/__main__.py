from pathlib import Path
import sys


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    project_dir = Path(__file__).resolve().parents[1]
    if arguments:
        from .cli import main as run_command
        return run_command(arguments, project_dir)
    from .app import launch_desktop_app
    launch_desktop_app(project_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
