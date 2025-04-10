import subprocess
import time
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import re

ALGO_PATH = "round2/iter1.py"
PROFIT_LOG = "profit_log2.txt"

space = [
    ## Old Parameters for round 1
    # Real(-1.0, 0.0, name="reversion_beta"),
    # Real(0, 3, name="take_width"),
    # Real(0, 3, name="clear_width"),
    # Integer(5, 25, name="adverse_volume"),

    # Real(0, 4, name="disregard_edge"),
    # Real(0, 10, name="join_edge"),
    # Real(0, 1, name="default_edge"),

    # Integer(5, 5000, name="z_rolling_window"),
    # Real(0, 10, name="zscore_threshold"),


    # # Basket-level
    # Real(0, 20, name="PICNIC_BASKET1.spread_threshold"),
    # Integer(1, 60, name="PICNIC_BASKET1.max_trade_size"),

    # Real(0, 20, name="PICNIC_BASKET2.spread_threshold"),
    # Integer(1, 100, name="PICNIC_BASKET2.max_trade_size"),

    # CROISSANTS
    Real(0, 3, name="CROISSANTS.take_width"),
    Real(0, 3, name="CROISSANTS.clear_width"),
    Integer(5, 250, name="CROISSANTS.adverse_volume"),
    Real(0, 4, name="CROISSANTS.disregard_edge"),
    Real(0, 10, name="CROISSANTS.join_edge"),
    Real(0, 2, name="CROISSANTS.default_edge"),

    # # JAMS
    # Real(0, 3, name="JAMS.take_width"),
    # Real(0, 3, name="JAMS.clear_width"),
    # Integer(5, 350, name="JAMS.adverse_volume"),
    # Real(0, 4, name="JAMS.disregard_edge"),
    # Real(0, 10, name="JAMS.join_edge"),
    # Real(0, 2, name="JAMS.default_edge"),

    # # DJEMBES
    # Real(0, 3, name="DJEMBES.take_width"),
    # Real(0, 3, name="DJEMBES.clear_width"),
    # Integer(5, 60, name="DJEMBES.adverse_volume"),
    # Real(0, 4, name="DJEMBES.disregard_edge"),
    # Real(0, 10, name="DJEMBES.join_edge"),
    # Real(0, 2, name="DJEMBES.default_edge"),
]



# Objective function: we NEGATE profit because skopt does minimization
@use_named_args(space)
def objective(**params):
    # Clear profit log before each run
    open(PROFIT_LOG, "w").close()

    # Build the command for round 1
    # command = [
    #     "prosperity3bt ",
    #     ALGO_PATH,
    #     f"--reversion_beta", str(params["reversion_beta"]),
    #     f"--take_width", str(params["take_width"]),
    #     f"--clear_width", str(params["clear_width"]),
    #     f"--adverse_volume", str(params["adverse_volume"]),
    #     f"--disregard_edge", str(params["disregard_edge"]),
    #     f"--join_edge", str(params["join_edge"]),
    #     f"--default_edge", str(params["default_edge"]),
    #     f"--z_rolling_window", str(params["z_rolling_window"]),
    #     f"--zscore_threshold", str(params["zscore_threshold"]),
    #     "1--1",
    #     "1--2"
    # ]

    # Build the command for round 2
    command = [
        "prosperity3bt ",
        ALGO_PATH,
        # f"--PICNIC_BASKET1.spread_threshold", str(params["PICNIC_BASKET1.spread_threshold"]),
        # f"--PICNIC_BASKET1.max_trade_size", str(params["PICNIC_BASKET1.max_trade_size"]),

        # f"--PICNIC_BASKET2.spread_threshold", str(params["PICNIC_BASKET2.spread_threshold"]),
        # f"--PICNIC_BASKET2.max_trade_size", str(params["PICNIC_BASKET2.max_trade_size"]),

        f"--CROISSANTS.take_width", str(params["CROISSANTS.take_width"]),
        f"--CROISSANTS.clear_width", str(params["CROISSANTS.clear_width"]),
        f"--CROISSANTS.adverse_volume", str(params["CROISSANTS.adverse_volume"]),
        f"--CROISSANTS.disregard_edge", str(params["CROISSANTS.disregard_edge"]),
        f"--CROISSANTS.join_edge", str(params["CROISSANTS.join_edge"]),
        f"--CROISSANTS.default_edge", str(params["CROISSANTS.default_edge"]),

        # f"--JAMS.take_width", str(params["JAMS.take_width"]),
        # f"--JAMS.clear_width", str(params["JAMS.clear_width"]),
        # f"--JAMS.adverse_volume", str(params["JAMS.adverse_volume"]),
        # f"--JAMS.disregard_edge", str(params["JAMS.disregard_edge"]),
        # f"--JAMS.join_edge", str(params["JAMS.join_edge"]),
        # f"--JAMS.default_edge", str(params["JAMS.default_edge"]),

        # f"--DJEMBES.take_width", str(params["DJEMBES.take_width"]),
        # f"--DJEMBES.clear_width", str(params["DJEMBES.clear_width"]),
        # f"--DJEMBES.adverse_volume", str(params["DJEMBES.adverse_volume"]),
        # f"--DJEMBES.disregard_edge", str(params["DJEMBES.disregard_edge"]),
        # f"--DJEMBES.join_edge", str(params["DJEMBES.join_edge"]),
        # f"--DJEMBES.default_edge", str(params["DJEMBES.default_edge"]),
        "2--1",
        "2-0"
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
for name, val in zip(["spread_threshold", "max_trade_size"], result.x):
    print(f"  {name}: {val}")

print(f"\n Best Total Profit: {-result.fun}")
