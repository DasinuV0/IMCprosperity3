import json
from typing import Any, List, Dict, Union
import jsonpickle

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Trader:
    def calculate_orders(self, product: str, order_depth: OrderDepth, current_position: int, 
                         position_limit: int, fair_price: float, delta: float) -> List[Order]:
        orders = []
        
        # calculate available capacity
        buy_capacity = position_limit - current_position  # how many more we can buy
        sell_capacity = position_limit + current_position  # how many more we can sell
        volume = 1

        # BUY logic
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_volume = order_depth.sell_orders[best_ask]
            
            if best_ask < fair_price:
                print(f"{product} BUY {best_ask_volume}x {best_ask} (Position: {current_position}, Capacity: {buy_capacity})")
                orders.append(Order(product, best_ask, -best_ask_volume))
        
        # SELL logic
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_volume = order_depth.buy_orders[best_bid]
            
            if best_bid > fair_price:
                print(f"{product} SELL {-best_bid_volume}x {best_bid} (Position: {current_position}, Capacity: {sell_capacity})")
                orders.append(Order(product, best_bid, -best_bid_volume))
        
        # Market make
        buy_price = fair_price - delta
        sell_price = fair_price + delta
        if buy_capacity > 0:
            print(f"{product} MARKET MAKE BUY {volume}x {buy_price} (Position: {current_position}, Capacity: {buy_capacity})")
            orders.append(Order(product, buy_price, volume))
        
            print(f"{product} MARKET MAKE SELL {-volume}x {sell_price} (Position: {current_position}, Capacity: {sell_capacity})")
            orders.append(Order(product, sell_price, -volume))
        elif sell_capacity > 0:
            print(f"{product} MARKET MAKE SELL {-volume}x {sell_price} (Position: {current_position}, Capacity: {sell_capacity})")
            orders.append(Order(product, sell_price, -volume))
        
            print(f"{product} MARKET MAKE BUY {volume}x {buy_price} (Position: {current_position}, Capacity: {buy_capacity})")
            orders.append(Order(product, buy_price, volume))

        return orders

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 1
        positions_summary = {}
        
        # define trading configurations for each product inside the run function
        trading_config = {
            'RAINFOREST_RESIN': {
                'position_limit': 50,
                'fair_price': 10000,
                'delta': 1
            }
        }
        
        # process each product
        for product in state.order_depths.keys():
            if product in trading_config:
                # get the product's order depth and configuration
                order_depth = state.order_depths[product]
                current_position = state.position.get(product, 0)
                config = trading_config[product]
                
                # store position for logging
                positions_summary[product] = current_position
                
                # calculate orders based on the product's configuration
                orders = self.calculate_orders(
                    product,
                    order_depth,
                    current_position,
                    config['position_limit'],
                    config['fair_price'],
                    config['delta']
                )
                
                # store the orders in the result
                if orders:
                    result[product] = orders
        
        # store useful state information for next round
        trader_data = {
            "timestamp": state.timestamp,
            "positions": positions_summary,
            "notes": "Modular trading logic for multiple products"
        }
        
        trader_data_serialized = jsonpickle.encode(trader_data)
        
        return result, conversions, trader_data_serialized
