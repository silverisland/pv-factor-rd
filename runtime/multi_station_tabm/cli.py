from __future__ import annotations

import argparse
from pathlib import Path

from .api import evaluate, predict, train


FINAL_CONFIRMATION = "RUN_SEALED_FINAL_TEST"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Multi-station shared TabM runtime")
    commands = result.add_subparsers(dest="command", required=True)
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--data")
    train_parser.add_argument("--seed", type=int)
    train_parser.add_argument("--factor", action="append", dest="factor_ids")

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--data")
    evaluate_parser.add_argument("--period", choices=["confirmation", "final_test"], required=True)
    evaluate_parser.add_argument("--confirm")
    evaluate_parser.add_argument("--output")
    evaluate_parser.add_argument("--seed", type=int)
    evaluate_parser.add_argument("--factor", action="append", dest="factor_ids")

    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--config", required=True)
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--data", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--seed", type=int)
    predict_parser.add_argument("--factor", action="append", dest="factor_ids")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "train":
        result = train(
            args.config, args.data, seed=args.seed, factor_ids=args.factor_ids
        )
        print(result["training_summary"].to_string(index=False))
        print(result["validation_metrics"].to_string(index=False))
        print(result["station_macro_summary"].to_string(index=False))
        print(result["checkpoint_dir"])
        return
    if args.command == "evaluate":
        if args.period == "final_test" and args.confirm != FINAL_CONFIRMATION:
            raise SystemExit(f"Final test is sealed; pass --confirm {FINAL_CONFIRMATION}")
        result = evaluate(
            args.checkpoint,
            args.config,
            period_name=args.period,
            data=args.data,
            seed=args.seed,
            factor_ids=args.factor_ids,
        )
        print(result["by_horizon"].to_string(index=False))
        print(result["by_horizon_group"].to_string(index=False))
        if args.output:
            output = Path(args.output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            result["predictions"].to_parquet(output, index=False)
            result["by_horizon"].to_csv(output.with_suffix(".by_horizon.csv"), index=False)
            result["by_station"].to_csv(
                output.with_suffix(".by_station.csv"), index=False
            )
            result["station_macro_summary"].to_csv(
                output.with_suffix(".station_macro_summary.csv"), index=False
            )
            result["by_day"].to_csv(output.with_suffix(".by_day.csv"), index=False)
            result["by_month"].to_csv(output.with_suffix(".by_month.csv"), index=False)
            result["monthly_score_summary"].to_csv(
                output.with_suffix(".monthly_score_summary.csv"), index=False
            )
            result["by_horizon_group"].to_csv(
                output.with_suffix(".by_horizon_group.csv"), index=False
            )
            result["audit"].to_csv(output.with_suffix(".audit.csv"), index=False)
        return
    result = predict(
        args.checkpoint,
        args.config,
        args.data,
        seed=args.seed,
        factor_ids=args.factor_ids,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(output)


if __name__ == "__main__":
    main()
