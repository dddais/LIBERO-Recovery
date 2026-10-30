"""Write a dedicated LIBERO config without touching ~/.libero or importing the simulator."""

import argparse
from pathlib import Path

from .protocol import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory to use as LIBERO_CONFIG_PATH")
    args = parser.parse_args()
    base = ROOT / "LIBERO/libero/libero"
    paths = {"benchmark_root": base, "bddl_files": base / "bddl_files", "init_states": base / "init_files",
             "assets": base / "assets", "datasets": ROOT / "datasets"}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "config.yaml").open("x") as stream:
        for key, value in paths.items():
            stream.write(f"{key}: {str(value)!r}\n")


if __name__ == "__main__":
    main()
