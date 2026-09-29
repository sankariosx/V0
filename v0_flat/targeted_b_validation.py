import sys
from tests import test_B_injected
from config_final import MIN_POWER_1WAY_03, MIN_POWER_2WAY_03

results = test_B_injected(
    effect_sizes=[0.3],
    n_per_size=10,
    effect_types=("1way", "2way"),
    seed_base=2000,
)

p1 = results["1way"][0.3]["power"]
p2 = results["2way"][0.3]["power"]

print(f"TARGETED B1 POWER @0.3: {p1:.2f} (required >= {MIN_POWER_1WAY_03:.2f})")
print(f"TARGETED B2 POWER @0.3: {p2:.2f} (required >= {MIN_POWER_2WAY_03:.2f})")

if p1 < MIN_POWER_1WAY_03 or p2 < MIN_POWER_2WAY_03:
    print("TARGETED VALIDATION: FAIL — do not start the 170-job validation.")
    sys.exit(1)

print("TARGETED VALIDATION: PASS — B1 and B2 meet the locked power thresholds.")


