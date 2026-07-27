"""Extract token values from aff_token_names and plot their frequency distribution."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_tokens(cell) -> list[str]:
    """Parse one aff_token_names cell (JSON list of dicts) and return token strings."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, list):
        items = cell
    else:
        text = str(cell).strip()
        if not text:
            return []
        items = json.loads(text)
    tokens: list[str] = []
    for item in items:
        if isinstance(item, dict) and "token" in item:
            tokens.append(str(item["token"]))
    return tokens


def display_token(token: str) -> str:
    """Make BPE space marker (Ġ) readable on the plot."""
    return token.replace("\u0120", "␣").replace("Ġ", "␣")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot distribution of token values in aff_token_names."
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path(r"x:\Yiqian\Downloads\validation_samples.xlsx"),
        help="Path to validation_samples.xlsx",
    )
    parser.add_argument(
        "--column",
        default="aff_token_names",
        help="Column name containing JSON token list",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Show top-k most frequent tokens (0 = all)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"x:\Yiqian\Downloads\aff_token_distribution.png"),
        help="Output figure path",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path(r"x:\Yiqian\Downloads\aff_token_counts.csv"),
        help="Optional CSV of token counts",
    )
    args = parser.parse_args()

    df = pd.read_excel(args.excel)
    if args.column not in df.columns:
        raise KeyError(f"Column '{args.column}' not found. Available: {list(df.columns)}")

    all_tokens: list[str] = []
    empty_cells = 0
    parse_errors = 0
    for cell in df[args.column]:
        if cell is None or (isinstance(cell, float) and pd.isna(cell)):
            empty_cells += 1
            continue
        try:
            tokens = parse_tokens(cell)
            if not tokens:
                empty_cells += 1
            all_tokens.extend(tokens)
        except (json.JSONDecodeError, TypeError, ValueError):
            parse_errors += 1

    counter = Counter(all_tokens)
    if not counter:
        raise RuntimeError("No tokens extracted. Check column content / JSON format.")

    counts_df = (
        pd.DataFrame(counter.most_common(), columns=["token", "count"])
        .assign(display=lambda d: d["token"].map(display_token))
    )

    plot_df = counts_df if args.top_k <= 0 else counts_df.head(args.top_k)

    fig_h = max(6.0, 0.28 * len(plot_df))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(plot_df["display"][::-1], plot_df["count"][::-1], color="#4C78A8")
    ax.set_xlabel("Count")
    ax.set_ylabel("token")
    title_k = "all" if args.top_k <= 0 else f"top {args.top_k}"
    ax.set_title(f"aff_token_names token distribution ({title_k})")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    csv_saved = True
    try:
        counts_df.to_csv(args.csv_out, index=False, encoding="utf-8-sig")
    except PermissionError:
        csv_saved = False

    counts = list(counter.values())
    print("=== Statistics ===")
    print(f"Excel rows          : {len(df)}")
    print(f"Empty / no-token    : {empty_cells}")
    print(f"Parse errors        : {parse_errors}")
    print(f"Tokens extracted    : {len(all_tokens)}")
    print(f"Unique tokens       : {len(counter)}")
    print(f"Max count           : {max(counts)}")
    print(f"Min count           : {min(counts)}")
    print(f"Mean count          : {sum(counts) / len(counts):.2f}")
    print(f"Saved figure        : {args.out}")
    if csv_saved:
        print(f"Saved counts CSV    : {args.csv_out}")
    else:
        print(f"Saved counts CSV    : FAILED (file locked?): {args.csv_out}")
    print("=== Top 15 tokens ===")
    for token, count in counter.most_common(15):
        # Prefer ASCII for Windows consoles (GBK); keep raw token otherwise.
        shown = token.replace("\u0120", " ").replace("Ġ", " ")
        try:
            print(f"  {shown!r}: {count}")
        except UnicodeEncodeError:
            print(f"  {token.encode('unicode_escape').decode()}: {count}")


if __name__ == "__main__":
    main()
