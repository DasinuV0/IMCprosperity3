import json
from typing import Any, List, Dict, Union
import jsonpickle

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Trader:
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 1
        trader_data = ""

        for product in state.order_depths.keys():
            if product == 'RAINFOREST_RESIN':
                order_depth: OrderDepth = state.order_depths[product]
                orders: List[Order] = []
                acceptable_price = 10000
                print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

                # BUY logic
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    if best_ask < acceptable_price:
                        print("BUY", str(-best_ask_volume) + "x", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))

                # SELL logic
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    if best_bid > acceptable_price:
                        print("SELL", str(best_bid_volume) + "x", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))

                result[product] = orders

        # You can customize this to track useful state info
        trader_data = {
            "timestamp": state.timestamp,
            "positions": state.position.get('RAINFOREST_RESIN', 0),
            "notes": "Basic trading logic for RAINFOREST_RESIN"
        }

        trader_data_serialized = jsonpickle.encode(trader_data)

        return result, conversions, trader_data_serialized
