"""Run the complete system pipeline with a managed FastAPI model server.

Example:
    python -m src.jobs.run_system_pipeline --market-start-date 2024-01-01

Any unknown CLI options are forwarded to src.jobs.run_full_pipeline.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import requests
from dotenv import load_dotenv

from src.utils.logging_config import setup_logging

load_dotenv()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_command(command: Sequence[str], step_name: str, env: dict[str, str] | None = None) -> None:
    logger.info("=" * 90)
    logger.info("START STEP: %s", step_name)
    logger.info("COMMAND: %s", " ".join(command))

    result = subprocess.run(
        list(command),
        cwd=PROJECT_ROOT,
        env=env or os.environ.copy(),
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed step={step_name} returncode={result.returncode}")

    logger.info("DONE STEP: %s", step_name)


def model_api_is_healthy(api_url: str) -> bool:
    try:
        response = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def start_model_api(
    api_host: str,
    api_port: int,
    startup_timeout_seconds: int,
    env: dict[str, str],
) -> tuple[subprocess.Popen | None, object | None]:
    """Start uvicorn unless a compatible model API is already healthy."""
    api_url = f"http://{api_host}:{api_port}"
    if model_api_is_healthy(api_url):
        logger.info("Model API already healthy at %s", api_url)
        return None, None

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "model_api.log", "a", encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.inference.api_server:app",
        "--host",
        api_host,
        "--port",
        str(api_port),
    ]
    logger.info("Starting model API: %s", " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + startup_timeout_seconds
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "Model API exited before becoming healthy. "
                f"Check logs/model_api.log. returncode={process.returncode}"
            )
        if model_api_is_healthy(api_url):
            logger.info("Model API is healthy at %s", api_url)
            return process, log_file
        time.sleep(2)

    process.terminate()
    raise RuntimeError(
        f"Timed out waiting for model API at {api_url}. "
        "Check SENTIMENT_MODEL_PATH and logs/model_api.log."
    )


def stop_model_api(process: subprocess.Popen | None, log_file: object | None) -> None:
    if process is not None and process.poll() is None:
        logger.info("Stopping managed model API...")
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    if log_file is not None:
        log_file.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start Docker/PostgreSQL, start the FastAPI model server, "
            "then run the full market sentiment pipeline."
        )
    )
    parser.add_argument("--api-host", default=os.getenv("MODEL_API_HOST") or "127.0.0.1")
    parser.add_argument("--api-port", type=int, default=int(os.getenv("MODEL_API_PORT") or "8000"))
    parser.add_argument("--startup-timeout-seconds", type=int, default=300)
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--skip-api-start", action="store_true")
    parser.add_argument("--keep-api-running", action="store_true")
    parser.add_argument(
        "--no-market-validation",
        action="store_true",
        help="Do not add --run-market-validation to the forwarded full pipeline command.",
    )
    return parser


def main() -> None:
    setup_logging()
    parser = build_parser()
    args, pipeline_args = parser.parse_known_args()

    api_url = f"http://{args.api_host}:{args.api_port}"
    env = os.environ.copy()
    env["SENTIMENT_MODEL_API_URL"] = api_url
    env["MODEL_INFERENCE_BACKEND"] = "api"
    env["PYTHONIOENCODING"] = "utf-8"

    if not args.skip_docker:
        run_command(["docker", "compose", "up", "-d"], "docker_compose_up", env=env)

    process = None
    log_file = None
    try:
        if not args.skip_api_start:
            process, log_file = start_model_api(
                api_host=args.api_host,
                api_port=args.api_port,
                startup_timeout_seconds=args.startup_timeout_seconds,
                env=env,
            )
        elif not model_api_is_healthy(api_url):
            raise RuntimeError(f"--skip-api-start was set, but {api_url}/health is not healthy.")

        full_pipeline_cmd = [
            sys.executable,
            "-m",
            "src.jobs.run_full_pipeline",
            "--model-api-url",
            api_url,
            "--inference-backend",
            "api",
        ]
        if not args.no_market_validation and "--run-market-validation" not in pipeline_args:
            full_pipeline_cmd.append("--run-market-validation")
        full_pipeline_cmd.extend(pipeline_args)

        run_command(full_pipeline_cmd, "run_full_pipeline", env=env)
    finally:
        if not args.keep_api_running:
            stop_model_api(process, log_file)


if __name__ == "__main__":
    main()
