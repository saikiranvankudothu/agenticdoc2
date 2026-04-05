"""
run_semantic.py — CLI runner for Step 5
========================================
Usage:
    python run_semantic.py --input output/paper/json/layout.json
    python run_semantic.py --input output/paper/json/layout.json --output output/paper/json/semantic.json
"""

import argparse
import json
import logging
from pathlib import Path

from agents.semantic_understanding_agent import SemanticUnderstandingAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5 — Semantic Understanding Agent")
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to layout.json from Layout Detection Agent (Step 4)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path (default: <input_stem>_semantic.json)")
    parser.add_argument("--model",  default="all-MiniLM-L6-v2",
                        help="SBERT model name")
    parser.add_argument("--beta",   type=float, default=0.7,
                        help="β for Equation 4 (default: 0.7)")
    parser.add_argument("--top-k",  type=int, default=3,
                        help="Top-K figure links per caption (default: 3)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = (
        Path(args.output) if args.output
        else input_path.parent / f"{input_path.stem}_semantic.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    agent   = SemanticUnderstandingAgent(model_name=args.model, beta=args.beta)
    results = agent.process_from_layout_json(str(input_path), top_k_links=args.top_k)

    output  = SemanticUnderstandingAgent.serialize(
        results["semantic_regions"], results["multimodal_links"]
    )
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logging.getLogger(__name__).info(
        "Saved %d regions, %d links → %s",
        len(results["semantic_regions"]),
        len(results["multimodal_links"]),
        output_path,
    )


if __name__ == "__main__":
    main()
