import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("AGG")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a PSNR/loss history saved as a pickle list.")
    parser.add_argument("input", type=Path, help="path to a .pkl history file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="output image path")
    parser.add_argument("--title", default="Training Curve", help="figure title")
    parser.add_argument("--ylabel", default="PSNR", help="y axis label")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output or args.input.with_suffix(".png")

    with args.input.open("rb") as f:
        history = pickle.load(f)

    if isinstance(history, dict):
        raise TypeError("Expected a list-like history, got a dict. Use the metric rank files for evaluation details.")

    plt.figure(figsize=(9, 5))
    plt.plot(range(len(history)), history, linewidth=2)
    plt.title(args.title)
    plt.xlabel("Epoch")
    plt.ylabel(args.ylabel)
    plt.grid(alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", dpi=160)
    plt.close()
    print(f"Saved curve to {output}")


if __name__ == "__main__":
    main()
