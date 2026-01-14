import argparse
import runpy
import sys


def _run(script: str) -> None:
    runpy.run_path(script, run_name="__main__")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ASL landmark pipeline")
    parser.add_argument(
        "mode",
        nargs="?",
        default="realtime",
        choices=["realtime", "extract", "train", "collect"],
        help="What to run (default: realtime)",
    )
    args = parser.parse_args(argv)

    if args.mode == "realtime":
        _run("realtime_asl_landmark_mlp.py")
    elif args.mode == "extract":
        _run("extract_landmarks_from_asl_alphabet.py")
    elif args.mode == "train":
        _run("train_asl_landmark_mlp.py")
    elif args.mode == "collect":
        _run("collect_landmark_csv.py")
    else:
        parser.error(f"Unknown mode: {args.mode}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
