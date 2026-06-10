"""Run market data + validation pipeline."""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date, timedelta

from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def run_command(command: list[str], step_name: str) -> None:
    logger.info("=" * 80)
    logger.info("START STEP: %s", step_name)
    logger.info("COMMAND: %s", " ".join(command))

    result = subprocess.run(command, check=False)

    if result.returncode != 0:
        raise RuntimeError(f"Failed step={step_name} returncode={result.returncode}")

    logger.info("DONE STEP: %s", step_name)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--limit-tickers", type=int, default=None)
    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-report", action="store_true")

    args = parser.parse_args()

    end_date = args.end_date or date.today().isoformat()
    start_date = args.start_date or (date.today() - timedelta(days=365)).isoformat()

    if not args.skip_fetch:
        cmd = [
            sys.executable,
            "-m",
            "src.jobs.fetch_market_data",
            "--start-date",
            start_date,
            "--end-date",
            end_date,
        ]

        if args.tickers:
            cmd.extend(["--tickers", args.tickers])

        if args.limit_tickers:
            cmd.extend(["--limit-tickers", str(args.limit_tickers)])

        run_command(cmd, "fetch_market_data")

    run_command(
        [sys.executable, "-m", "src.jobs.build_market_features"],
        "build_market_features",
    )

    if not args.skip_report:
        run_command(
            [sys.executable, "-m", "src.jobs.run_sentiment_validation", "--skip-import"],
            "run_sentiment_validation",
        )


if __name__ == "__main__":
    main()