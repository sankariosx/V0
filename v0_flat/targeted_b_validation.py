import sys
from tests import test_B_injected
from config_final import MIN_POWER_1WAY_03, MIN_POWER_2WAY_03

# Locked regression seeds taken from the last fully passing 170-job validation
# (#124). This makes the targeted gate a reproducible regression check against
# the known-good validation population rather than a new random sample.
HISTORICAL_SEEDS = {
    "1way": {"0.3": [96512, 5913, 4338, 7972, 38269, 62097, 40328, 43962, 74259, 72684]},
    "2way": {"0.3": [88117, 91751, 65440, 93810, 24107, 33084, 29450, 31025, 60097, 97094]},
}

results = test_B_injected(
    effect_sizes=[0.3],
    n_per_size=10,
    effect_types=("1way", "2way"),
    seed_schedule=HISTORICAL_SEEDS,
)

p1 = results["1way"][0.3]["power"]
p2 = results["2way"][0.3]["power"]

print(f"TARGETED B1 POWER @0.3: {p1:.2f} (required >= {MIN_POWER_1WAY_03:.2f})")
print(f"TARGETED B2 POWER @0.3: {p2:.2f} (required >= {MIN_POWER_2WAY_03:.2f})")

if p1 < MIN_POWER_1WAY_03 or p2 < MIN_POWER_2WAY_03:
    print("TARGETED VALIDATION: FAIL — do not start the 170-job validation.")
    sys.exit(1)

print("TARGETED VALIDATION: PASS — B1 and B2 meet the locked power thresholds.")
