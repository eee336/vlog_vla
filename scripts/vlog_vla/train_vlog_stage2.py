import sys

from train_common import run_stage_from_cli


if __name__ == "__main__":
    if "--stage" not in sys.argv:
        sys.argv.extend(["--stage", "2"])
    run_stage_from_cli()
