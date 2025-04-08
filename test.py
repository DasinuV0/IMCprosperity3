import os
import sys
import json
import re
import subprocess
import math
import numpy as np
import jsonpickle
from typing import List
from datamodel import OrderDepth, UserId, TradingState, Order
from logger import Logger

logging = Logger()

# -------------------------------
# Configuration file management
# -------------------------------
CONFIG_FILENAME = "trader_config.json"

def load_config():
    """Load configuration parameters from a JSON file, if available."""
    if os.path.exists(CONFIG_FILENAME):
        with open(CONFIG_FILENAME, "r") as f:
            return json.load(f)
    return None

def write_config(params):
    """Write configuration parameters to a JSON file."""
    with open(CONFIG_FILENAME, "w") as f:
        json.dump(params, f)

# -------------------------------
# Product & Default Parameters
# -------------------------------
class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    KELP = "KELP"

# Default parameters for each product.
# (These remain unchanged for RAINFOREST_RESIN.)
DEFAULT_PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,  # for joining/pennying near fair value
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.129,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.529,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
}

# -------------------------------
# Trader Class
# -------------------------------
class Trader:
    def __init__(self, params=None):
        # Attempt to load configuration from file regardless of how the Trader is called.
        loaded_config = load_config()
        if loaded_config is not None:
            # If a config is loaded, use it for SQUID_INK and KELP,
            # while always using DEFAULT_PARAMS for RAINFOREST_RESIN.
            params = {
                Product.RAINFOREST_RESIN: DEFAULT_PARAMS[Product.RAINFOREST_RESIN],
                Product.SQUID_INK: DEFAULT_PARAMS[Product.SQUID_INK],
                Product.KELP: loaded_config,
            }
        elif params is None:
            # If no configuration is provided or found,
            # use the default parameters.
            params = DEFAULT_PARAMS.copy()
        self.params = params
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 45,
            Product.SQUID_INK: 45,
            Product.KELP: 45
        }

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(
                volume for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def squid_ink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.params[Product.SQUID_INK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                mmmid_price = ((best_ask + best_bid) / 2
                               if traderObject.get("squid_ink_last_price", None) is None
                               else traderObject["squid_ink_last_price"])
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            if traderObject.get("squid_ink_last_price", None) is not None:
                last_price = traderObject["squid_ink_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[Product.SQUID_INK]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["squid_ink_last_price"] = mmmid_price
            return fair
        return None

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                mmmid_price = ((best_ask + best_bid) / 2
                               if traderObject.get("kelp_last_price", None) is None
                               else traderObject["kelp_last_price"])
            else:
                mmmid_price = (mm_ask + mm_bid) / 2
            if traderObject.get("kelp_last_price", None) is not None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params[Product.KELP]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width,
            orders, order_depth, position,
            buy_order_volume, sell_order_volume,
            prevent_adverse, adverse_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width,
            orders, order_depth, position,
            buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(
        self,
        product,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        manage_position: bool = False,
        soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        asks_above_fair = [price for price in order_depth.sell_orders.keys()
                           if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys()
                           if price < fair_value - disregard_edge]
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None
        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1
        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1
        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position,
            buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result = {}
        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = (state.position[Product.RAINFOREST_RESIN]
                              if Product.RAINFOREST_RESIN in state.position else 0)
            resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["take_width"],
                resin_position,
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                self.params[Product.RAINFOREST_RESIN]["clear_width"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
            )
            resin_make_orders, _, _ = self.make_orders(
                Product.RAINFOREST_RESIN,
                state.order_depths[Product.RAINFOREST_RESIN],
                self.params[Product.RAINFOREST_RESIN]["fair_value"],
                resin_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFOREST_RESIN]["disregard_edge"],
                self.params[Product.RAINFOREST_RESIN]["join_edge"],
                self.params[Product.RAINFOREST_RESIN]["default_edge"],
                True,
                self.params[Product.RAINFOREST_RESIN]["soft_position_limit"],
            )
            result[Product.RAINFOREST_RESIN] = (resin_take_orders +
                                               resin_clear_orders +
                                               resin_make_orders)
        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_ink_position = (state.position[Product.SQUID_INK]
                                  if Product.SQUID_INK in state.position else 0)
            squid_ink_fair_value = self.squid_ink_fair_value(
                state.order_depths[Product.SQUID_INK], traderObject
            )
            squid_ink_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_ink_fair_value,
                self.params[Product.SQUID_INK]["take_width"],
                squid_ink_position,
                self.params[Product.SQUID_INK]["prevent_adverse"],
                self.params[Product.SQUID_INK]["adverse_volume"],
            )
            squid_ink_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_ink_fair_value,
                self.params[Product.SQUID_INK]["clear_width"],
                squid_ink_position,
                buy_order_volume,
                sell_order_volume,
            )
            squid_ink_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_ink_fair_value,
                squid_ink_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                self.params[Product.SQUID_INK]["default_edge"],
            )
            result[Product.SQUID_INK] = (squid_ink_take_orders +
                                         squid_ink_clear_orders +
                                         squid_ink_make_orders)
        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (state.position[Product.KELP]
                             if Product.KELP in state.position else 0)
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["take_width"],
                kelp_position,
                self.params[Product.KELP]["prevent_adverse"],
                self.params[Product.KELP]["adverse_volume"],
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                self.params[Product.KELP]["clear_width"],
                kelp_position,
                buy_order_volume,
                sell_order_volume,
            )
            kelp_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                kelp_fair_value,
                kelp_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (kelp_take_orders +
                                    kelp_clear_orders +
                                    kelp_make_orders)
        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        logging.flush(state, result, conversions, traderData)
        return result, conversions, traderData

# -------------------------------
# Grid Search Logic
# -------------------------------
# Define the tuning grid for SQUID_INK and KELP.
param_grid = {
    "take_width": [0.5, 1, 1.5],
    "clear_width": [0, 1],
    "prevent_adverse": [True, False],
    "adverse_volume": [10, 15, 20],
    "reversion_beta": [-0.7, -0.529, -0.3],
    "disregard_edge": [0.5, 1, 2],
    "join_edge": [0, 0.5, 1],
    "default_edge": [0.5, 1, 2],
}

def generate_param_combinations(param_grid):
    keys = list(param_grid.keys())
    combinations = []
    def recurse(current, depth):
        if depth == len(keys):
            combinations.append(dict(zip(keys, current)))
            return
        for value in param_grid[keys[depth]]:
            recurse(current + [value], depth + 1)
    recurse([], 0)
    return combinations

param_combinations = generate_param_combinations(param_grid)

def run_simulation() -> float:
    """
    Runs the external backtest by invoking
      prosperity3bt test.py
    Waits for it to finish and parses the output to extract the Total profit.
    """
    try:
        proc = subprocess.run(
            ["prosperity3bt", "test.py", "1"],
            capture_output=True, text=True, timeout=30
        )
        output = proc.stdout
        profits = re.findall(r"Total profit:\s*([\d,]+)", output)
        if profits:
            profit_str = profits[-1].replace(",", "")
            return float(profit_str)
        else:
            return 0.0
    except Exception as e:
        print("Error running simulation:", e)
        return 0.0

def grid_search():
    best_score = -np.inf
    best_params = None
    for param_comb in param_combinations:
        # Build a parameter dictionary for SQUID_INK and KELP.
        config_params = {
            "take_width": param_comb["take_width"],
            "clear_width": param_comb["clear_width"],
            "prevent_adverse": param_comb["prevent_adverse"],
            "adverse_volume": param_comb["adverse_volume"],
            "reversion_beta": param_comb["reversion_beta"],
            "disregard_edge": param_comb["disregard_edge"],
            "join_edge": param_comb["join_edge"],
            "default_edge": param_comb["default_edge"],
        }
        # Write the current configuration so the external process can read it.
        write_config(config_params)
        # Run the external backtest.
        score = run_simulation()
        print("Params:", param_comb, "Score (Total Profit):", score)
        if score > best_score:
            best_score = score
            best_params = param_comb
    print("Best Score:", best_score)
    print("Best Parameters:", best_params)

# -------------------------------
# Main Routine
# -------------------------------
if __name__ == '__main__':
    # If "grid" is passed as an argument, run the grid search.
    if len(sys.argv) > 1 and sys.argv[1].lower() == "grid":
        grid_search()
