import subprocess
import time
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import re

ALGO_PATH = "Iteration5.py"
PROFIT_LOG = "profit_log.txt"

space = [
    Real(-1.0, 0.0, name="reversion_beta"),
    Real(0, 3, name="take_width"),
    Real(0, 3, name="clear_width"),
    Integer(5, 25, name="adverse_volume"),

    Real(5, 25, name="disregard_edge"),
    Real(5, 25, name="join_edge"),
    Real(5, 25, name="default_edge"),

    Integer(5, 25, name="z_rolling_window"),
    Real(0, 10, name="zscore_threshold"),
]

# Objective function: we NEGATE profit because skopt does minimization
@use_named_args(space)
def objective(**params):
    # Clear profit log before each run
    open(PROFIT_LOG, "w").close()

    # Build the command
    command = [
        "prosperity3submit ",
        ALGO_PATH,
        f"--reversion_beta", str(params["reversion_beta"]),
        f"--take_width", str(params["take_width"]),
        f"--clear_width", str(params["clear_width"]),
        f"--adverse_volume", str(params["adverse_volume"]),
        f"--disregard_edge", str(params["disregard_edge"]),
        f"--join_edge", str(params["join_edge"]),
        f"--default_edge", str(params["default_edge"]),
        f"--z_rolling_window", str(params["z_rolling_window"]),
        f"--zscore_threshold", str(params["zscore_threshold"]),
    ]

    try:
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, check=True)

        # Wait briefly to ensure log is written
        time.sleep(0.5)

        with open(PROFIT_LOG, "r") as f:
            last_line = f.readlines()[-1]

            match = re.search(r"Tot Profit:\s*([-\d.]+)", last_line)
            if not match:
                print("Profit line not found!")
                return 1e9  # Penalize

            profit = float(match.group(1))
            print(f"Profit: {profit}")
            return -profit  # Negate for maximization
        
    except Exception as e:
        print(f"Error during run: {e}")
        return 1e9  # Penalize failed runs

result = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=50,
    random_state=42,
    verbose=True
)

print("\n Best Parameters:")
for name, val in zip(["take_width","clear_width","prevent_adverse","adverse_volume","reversion_beta","disregard_edge","join_edge","default_edge","z_rolling_window","zscore_threshold"], result.x):
    print(f"  {name}: {val}")

print(f"\n Best Total Profit: {-result.fun}")
