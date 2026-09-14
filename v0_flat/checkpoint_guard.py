"""Validate V0 checkpoints without rewriting their provenance."""
import json
import sys
from pathlib import Path

import config_final as CFG

CHECKPOINT_DIR = CFG.RESULTS_DIR / "checkpoints"
EXPECTED_VALIDATION_VERSION = CFG.VALIDATION_VERSION
EXPECTED_FEATURE_VERSION = CFG.FEATURE_VERSION


def checkpoint_is_current(data):
    # Current runner checkpoints predate explicit code-version stamping, so
    # a missing validation_version is accepted ONLY when the feature/config
    # provenance is current. Any explicit older validation version is stale.
    validation_ok = data.get("validation_version") in (None, EXPECTED_VALIDATION_VERSION)
    return validation_ok and data.get("feature_version") == EXPECTED_FEATURE_VERSION


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
            print(
                "Checkpoint version mismatch "
                f"(validation found {data.get('validation_version', 'missing')}, expected {EXPECTED_VALIDATION_VERSION}; "
                f"feature found {data.get('feature_version', 'missing')}, expected {EXPECTED_FEATURE_VERSION}) "
                f"— discarding stale checkpoint {path.name}."
            )
            path.unlink(missing_ok=True)
            removed += 1
    print(
        f"Checkpoint guard: checked={checked}, discarded={removed}, "
        f"validation_version={EXPECTED_VALIDATION_VERSION}, feature_version={EXPECTED_FEATURE_VERSION}"
    )
    return removed


def stamp_checkpoints():
    # Never mutate old results to make them look like they were produced by
    # a newer statistical procedure. Verify only; the runner's config hash
    # remains the authoritative resume/aggregation key.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    verified = 0
    invalid = 0
    for path in sorted(CHECKPOINT_DIR.glob("*.json")):
        try:
            with path.open() as f:
                data = json.load(f)
        except Exception:
            invalid += 1
            continue
        if checkpoint_is_current(data):
            verified += 1
        else:
            invalid += 1
            print(f"Refusing to upload incompatible checkpoint: {path.name}")
    print(
        f"Checkpoint provenance check: verified={verified}, invalid={invalid}, "
        f"validation_version={EXPECTED_VALIDATION_VERSION}, feature_version={EXPECTED_FEATURE_VERSION}"
    )
    if invalid:
        raise SystemExit(f"Refusing checkpoint upload: {invalid} incompatible checkpoint(s).")
    return verified


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"guard", "stamp"}:
        raise SystemExit("Usage: python3 checkpoint_guard.py [guard|stamp]")
    if sys.argv[1] == "guard":
        guard_checkpoints()
    else:
        stamp_checkpoints()
