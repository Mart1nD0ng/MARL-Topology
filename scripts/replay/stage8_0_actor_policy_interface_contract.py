"""Print the Stage 8.0 actor policy interface contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from marl_topology.policies import build_stage8_0_actor_policy_interface_contract


def main() -> None:
    contract = build_stage8_0_actor_policy_interface_contract()
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
