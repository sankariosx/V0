"""Validate and stamp V0 validation checkpoints.

This guard prevents checkpoints produced by incompatible validation/feature
logic from being resumed or aggregated. It is intentionally small and is
called by the GitHub Actions workflow before validation and before checkpoint
artifacts are uploaded.
"""
import json
import sys
from pathlib import Path

import config_final as CFG

CHECKPOINT_DIR = CFG.RESULTS_DIR / "checkpoints"
EXPECTED_VALIDATION_VERSION = CFG.VALIDATION_VERSION
EXPECTED_FEATURE_VERSION = CFG.FEATURE_VERSION


def checkpoint_is_current(data):
    return (
        data.get("validation_version") == EXPECTED_VALIDATION_VERSION
        and data.get("feature_version") == EXPECTED_FEATURE_VERSION
    )


def guard_checkpoints():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    checked = 0
    for path in sorted(CHECKPOINT_DIR.glob("*.json")):
        try:
            with path.open() as f:
                data = json.load(f)
        except Exception:
            print(f"Invalid checkpoint {path.name} — discarding it.")
            path.unlink(missing_ok=True)
            removed += 1
            continue

        checked += 1
        if not checkpoint_is_current(data):
            found_validation = data.get("validation_version", "missing")
            found_feature = data.get("feature_version", "missing")
            print(
                "Checkpoint version mismatch "
                f"(validation found {found_validation}, expected {EXPECTED_VALIDATION_VERSION}; "
                f"feature found {found_feature}, expected {EXPECTED_FEATURE_VERSION}) "
                f"— discarding stale checkpoint {path.name} and starting fresh."
            )
            path.unlink(missing_ok=True)
            removed += 1

    print(
        f"Checkpoint guard: checked={checked}, discarded={removed}, "
        f"validation_version={EXPECTED_VALIDATION_VERSION}, "
        f"feature_version={EXPECTED_FEATURE_VERSION}"
    )
    return removed


def stamp_checkpoints():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    stamped = 0
    for path in sorted(CHECKPOINT_DIR.glob("*.json")):
        try:
            with path.open() as f:
                data = json.load(f)
        except Exception:
            print(f"Skipping invalid checkpoint during stamp: {path.name}")
            continue
        data["validation_version"] = EXPECTED_VALIDATION_VERSION
        data["feature_version"] = EXPECTED_FEATURE_VERSION
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
        stamped += 1
    print(
        f"Checkpoint stamp: stamped={stamped}, "
        f"validation_version={EXPECTED_VALIDATION_VERSION}, "
        f"feature_version={EXPECTED_FEATURE_VERSION}"
    )
    return stamped


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"guard", "stamp"}:
        raise SystemExit("Usage: python3 checkpoint_guard.py [guard|stamp]")
    if sys.argv[1] == "guard":
        guard_checkpoints()
    else:
        stamp_checkpoints()
