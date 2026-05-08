import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DCAS_RUNNER = Path("/home/leo/ads-skynet/DCAS-PolicyEngine/build/dcas_policy_runner")


@dataclass(frozen=True)
class DCASMockInput:
    attentive: bool = False
    reason: str = "drowsy"
    timestamp_ms: int = 1000
    reason_timestamp_ms: int | None = None
    delta_s: float = 2.5
    ticks: int = 1
    jetracer_input_0_4: float = 0.2
    lkas_throttle: float = 0.5
    lkas_mode: str = "ON_ACTIVE"
    switch_event: str = "NONE"
    driver_override: bool = False
    notebook_alive: bool = True


class DCASPolicyClient:
    """Thin subprocess adapter for the C++ DCAS policy runner."""

    def __init__(self, runner_path: str | os.PathLike[str] | None = None, timeout_s: float = 2.0):
        env_runner = os.environ.get("DCAS_POLICY_RUNNER")
        self.runner_path = Path(runner_path or env_runner or DEFAULT_DCAS_RUNNER)
        self.timeout_s = float(timeout_s)

    def ensure_runner_ready(self, auto_build: bool = False) -> None:
        """Validate runner availability and optionally build it."""
        if self.runner_path.exists():
            return

        if not auto_build:
            raise FileNotFoundError(
                f"DCAS runner not found: {self.runner_path}. "
                "Build it with: cmake --build /home/leo/ads-skynet/DCAS-PolicyEngine/build"
            )

        repo_dir = Path("/home/leo/ads-skynet/DCAS-PolicyEngine")
        build_dir = repo_dir / "build"
        subprocess.run(["cmake", "-S", str(repo_dir), "-B", str(build_dir)], check=True, timeout=120)
        subprocess.run(["cmake", "--build", str(build_dir)], check=True, timeout=240)

        if not self.runner_path.exists():
            raise FileNotFoundError(
                f"DCAS runner build completed but binary is missing: {self.runner_path}"
            )

    def preflight(self, auto_build: bool = False) -> None:
        """Run a lightweight sanity check for runner invocation."""
        self.ensure_runner_ready(auto_build=auto_build)
        subprocess.run(
            [str(self.runner_path), "--help"],
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )

    def evaluate(self, mock_input: DCASMockInput) -> list[dict[str, Any]]:
        self.ensure_runner_ready(auto_build=False)

        command = [
            str(self.runner_path),
            "--attentive",
            self._bool_arg(mock_input.attentive),
            "--reason",
            mock_input.reason,
            "--ts",
            str(mock_input.timestamp_ms),
            "--dt",
            str(mock_input.delta_s),
            "--ticks",
            str(mock_input.ticks),
            "--jetracer",
            str(mock_input.jetracer_input_0_4),
            "--lkas-throttle",
            str(mock_input.lkas_throttle),
            "--lkas-mode",
            mock_input.lkas_mode,
            "--switch",
            mock_input.switch_event,
            "--override",
            self._bool_arg(mock_input.driver_override),
            "--notebook-alive",
            self._bool_arg(mock_input.notebook_alive),
        ]

        if mock_input.reason_timestamp_ms is not None:
            command.extend(["--reason-ts", str(mock_input.reason_timestamp_ms)])

        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=self.timeout_s,
        )

        outputs: list[dict[str, Any]] = []
        for line in completed.stdout.splitlines():
            line = line.strip()
            if line:
                outputs.append(json.loads(line))
        return outputs

    @staticmethod
    def _bool_arg(value: bool) -> str:
        return "true" if value else "false"


def main() -> None:
    client = DCASPolicyClient()
    client.preflight(auto_build=False)
    outputs = client.evaluate(DCASMockInput())
    for output in outputs:
        print(json.dumps(output, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
