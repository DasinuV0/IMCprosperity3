from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import string
import jsonpickle
import numpy as np
import math

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    SQUID_INK = "SQUID_INK"
    KELP = "KELP"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBES = "DJEMBES"
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.SQUID_INK: {
        "take_width": 0.0,
        "clear_width": 0.7989037527218122,
        "prevent_adverse": True,
        "adverse_volume": 12,
        "reversion_beta": -0.262018206155453,
        "disregard_edge": 0.0,
        "join_edge": 8.205011595727127,
        "default_edge": 1.0,
        "z_rolling_window": 5000,     # number of ticks to consider for the rolling average
        "zscore_threshold": 3.1571449177636266,  # threshold for taking advantage of large swings
    },
    Product.KELP: {  # added KELP with similar settings as SQUID_INK
        "take_width": 3.0,
        "clear_width": 0.0,
        "prevent_adverse": True,
        "adverse_volume": 7,
        "reversion_beta": -0.231709901360756,
        "disregard_edge": 4.0,
        "join_edge": -1.8535170129297578,
        "default_edge": 1.0,
    },
    Product.CROISSANTS: {
        "take_width": 1.0,
        "clear_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 1,
    },
    Product.JAMS: {
        "take_width": 1.0,
        "clear_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 1,
    },
    Product.DJEMBES: {
        "take_width": 1.0,
        "clear_width": 0.5,
        "prevent_adverse": True,
        "adverse_volume": 10,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 1,
    },
    Product.PICNIC_BASKET1: {
        "spread_threshold": 50,
        "max_trade_size": 40,
    },
    Product.PICNIC_BASKET2: {
        "spread_threshold": 50,
        "max_trade_size": 40,
    }
}

BASKET_COMPOSITIONS = {
    Product.PICNIC_BASKET1: {
        Product.CROISSANTS: 6,
        Product.JAMS: 3,
        Product.DJEMBES: 1,
    },
    Product.PICNIC_BASKET2: {
        Product.CROISSANTS: 4,
        Product.JAMS: 2,
    }
}

class SimpleGARCH:
    """
    A simple GARCH(p,q) implementation without external dependencies.
    GARCH models time-varying volatility and is useful for predicting financial time series.
    """
    def __init__(self, p=1, q=1, omega=0.01, alpha=0.1, beta=0.8, history_size=100):
        """
        Initialize the GARCH model
        
        Parameters:
        p - order of GARCH terms (lagged conditional variances)
        q - order of ARCH terms (lagged squared returns)
        omega - constant term in variance equation
        alpha - coefficient for ARCH terms
        beta - coefficient for GARCH terms
        history_size - maximum history length to store
        """
        self.p = p  # GARCH order
        self.q = q  # ARCH order
        self.omega = omega  # constant in variance equation
        self.alpha = alpha  # ARCH coefficient
        self.beta = beta    # GARCH coefficient
        self.history_size = history_size
        
        # Price history
        self.price_history = []
        # Return history (percentage changes)
        self.return_history = []
        # Squared return history
        self.sq_return_history = []
        # Conditional variance history
        self.variance_history = []
        # Prediction history
        self.prediction_history = []
        
        # Initial variance estimate
        self.current_variance = omega / (1 - alpha - beta) if (alpha + beta) < 1 else 0.01
        
    def update(self, price):
        """Update the model with a new price observation"""
        # Store the price
        self.price_history.append(price)
        if len(self.price_history) > self.history_size:
            self.price_history = self.price_history[-self.history_size:]
            
        # Calculate return if we have at least 2 prices
        if len(self.price_history) >= 2:
            # Percentage return
            ret = (price - self.price_history[-2]) / self.price_history[-2]
            self.return_history.append(ret)
            if len(self.return_history) > self.history_size:
                self.return_history = self.return_history[-self.history_size:]
                
            # Squared return
            sq_ret = ret * ret
            self.sq_return_history.append(sq_ret)
            if len(self.sq_return_history) > self.history_size:
                self.sq_return_history = self.sq_return_history[-self.history_size:]
            
            # Update conditional variance using GARCH formula
            if len(self.sq_return_history) >= max(self.p, self.q) and len(self.variance_history) >= self.p:
                # ARCH component
                arch_component = sum(self.alpha * self.sq_return_history[-(i+1)] / self.q 
                                  for i in range(min(self.q, len(self.sq_return_history))))
                
                # GARCH component
                garch_component = sum(self.beta * self.variance_history[-(i+1)] / self.p 
                                     for i in range(min(self.p, len(self.variance_history))))
                
                # New conditional variance
                self.current_variance = self.omega + arch_component + garch_component
            else:
                # Initial estimate
                if len(self.sq_return_history) > 1:
                    self.current_variance = sum(self.sq_return_history) / len(self.sq_return_history)
            
            # Store the variance
            self.variance_history.append(self.current_variance)
            if len(self.variance_history) > self.history_size:
                self.variance_history = self.variance_history[-self.history_size:]
                
            # Make and store a prediction for evaluation
            prediction = self._predict_next_price()
            self.prediction_history.append(prediction)
            if len(self.prediction_history) > self.history_size:
                self.prediction_history = self.prediction_history[-self.history_size:]
    
    def _predict_next_price(self):
        """
        Predict the next price using a combination of:
        1. The current price
        2. A volatility-dependent adjustment
        """
        if len(self.price_history) < 2:
            return self.price_history[-1] if self.price_history else None
        
        current_price = self.price_history[-1]
        
        # Calculate short and long-term average returns
        if len(self.return_history) >= 5:
            short_term_avg = sum(self.return_history[-5:]) / 5
            long_term_avg = sum(self.return_history) / len(self.return_history)
            
            # Blend short and long term averages based on volatility
            # When volatility is high, favor short-term movements
            volatility_weight = min(5 * self.current_variance, 0.9)
            expected_return = (volatility_weight * short_term_avg + 
                              (1 - volatility_weight) * long_term_avg)
            
            # Apply volatility scaling - reduce expected return when volatility is high
            volatility_scaling = 1 / (1 + 10 * self.current_variance)
            adjusted_return = expected_return * volatility_scaling
            
            # Calculate predicted price
            predicted_price = current_price * (1 + adjusted_return)
            return predicted_price
        else:
            # Not enough history, use simple momentum
            if len(self.return_history) > 0:
                avg_return = sum(self.return_history) / len(self.return_history)
                return current_price * (1 + avg_return * 0.5)  # Dampen the effect
            else:
                return current_price
    
    def predict(self):
        """Predict the next price value"""
        if not self.price_history:
            return None
        
        return self._predict_next_price()
    
    def get_volatility(self):
        """Return the current volatility (standard deviation)"""
        return self.current_variance ** 0.5

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.SQUID_INK: 50,
            Product.KELP: 50,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
        }
        
        # Initialize GARCH model for SQUID_INK with carefully selected parameters
        self.squid_ink_garch = SimpleGARCH(
            p=1,                # GARCH term order
            q=1,                # ARCH term order
            omega=0.01,         # Constant in variance equation
            alpha=0.15,         # ARCH coefficient - impact of recent shocks
            beta=0.8,           # GARCH coefficient - persistence of volatility
            history_size=50     # Number of historical prices to store
        )

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
        buy_take_width: float = None,
        sell_take_width: float = None,
    ) -> (List[Order], int, int):
        position_limit = self.LIMIT[product]

        # Use asymmetric take_width if provided
        buy_width = buy_take_width if buy_take_width is not None else take_width
        sell_width = sell_take_width if sell_take_width is not None else take_width

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - buy_width:  # Use buy_width for buys
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
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
                if best_bid >= fair_value + sell_width:  # Use sell_width for sells
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return orders, buy_order_volume, sell_order_volume

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
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
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
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
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
            current_mid_price = (best_ask + best_bid) / 2
            
            # Update GARCH model with current mid price
            self.squid_ink_garch.update(current_mid_price)
            
            # Get GARCH prediction for next price
            garch_prediction = self.squid_ink_garch.predict()
            
            # Get current volatility from GARCH model
            volatility = self.squid_ink_garch.get_volatility()
            
            # Store price history for z-score calculation
            if "squid_ink_history" not in traderObject:
                traderObject["squid_ink_history"] = []
            traderObject["squid_ink_history"].append(current_mid_price)
            window = self.params[Product.SQUID_INK].get("z_rolling_window", 50)  # Shorter window for faster adaptation
            # Keep only the most recent 'window' values:
            if len(traderObject["squid_ink_history"]) > window:
                traderObject["squid_ink_history"] = traderObject["squid_ink_history"][-window:]
            
            # Calculate z-score and rolling mean
            hist = traderObject["squid_ink_history"]
            zscore = 0
            rolling_mean = current_mid_price  # Default to current price if not enough history
            
            if len(hist) >= 10:  # Need minimum data for reliable z-score
                rolling_mean = np.mean(hist)
                rolling_std = np.std(hist)
                if rolling_std == 0:
                    rolling_std = 1e-6  # avoid division by zero
                zscore = (current_mid_price - rolling_mean) / rolling_std
                traderObject["squid_ink_zscore"] = zscore
                
            # Original market microstructure-based calculation
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
            
            # Store volatility in trader object for potential use in strategy
            traderObject["squid_ink_volatility"] = volatility
            
            # Determine fair value by combining GARCH, mean reversion, and market microstructure
            if garch_prediction is not None:
                # Apply z-score based mean reversion to GARCH prediction
                threshold = self.params[Product.SQUID_INK].get("zscore_threshold", 2.0)
                reversion_beta = self.params[Product.SQUID_INK].get("reversion_beta", -0.2)
                
                # Only apply mean reversion if z-score exceeds threshold
                if abs(zscore) >= threshold:
                    # Calculate mean reversion adjustment
                    mean_reversion_adjustment = zscore * reversion_beta * volatility
                    # Apply stronger mean reversion in high volatility environments
                    garch_with_reversion = garch_prediction * (1 + mean_reversion_adjustment)
                else:
                    garch_with_reversion = garch_prediction
                    
                if mm_ask is not None and mm_bid is not None:
                    # Combine adjusted GARCH prediction with market microstructure
                    mm_mid = (mm_ask + mm_bid) / 2
                    
                    # Adjust weight based on volatility and z-score
                    # - High volatility: trust GARCH more (follows market trends)
                    # - High absolute z-score: trust mean reversion more
                    vol_scale = min(max(volatility * 10, 0.3), 0.8)
                    z_influence = min(abs(zscore) / threshold, 1.0) * 0.3  # Up to 30% weight to mean reversion
                    
                    # Blend the predictions
                    fair_value = (vol_scale * garch_with_reversion + 
                                 (1 - vol_scale - z_influence) * mm_mid + 
                                 z_influence * rolling_mean)  # Add some pull toward the mean
                else:
                    # Use adjusted GARCH prediction directly
                    fair_value = garch_with_reversion
            else:
                # Fall back to z-score based mean reversion if GARCH fails
                if "squid_ink_zscore" in traderObject:
                    zscore = traderObject["squid_ink_zscore"]
                    threshold = self.params[Product.SQUID_INK].get("zscore_threshold", 2.0)
                    reversion_beta = self.params[Product.SQUID_INK].get("reversion_beta", -0.2)
                    
                    if abs(zscore) >= threshold:
                        # Apply mean reversion
                        predicted_reversion = zscore * reversion_beta
                        fair_value = current_mid_price * (1 + predicted_reversion)
                    else:
                        # Use market microstructure or current mid price
                        if mm_ask is not None and mm_bid is not None:
                            fair_value = (mm_ask + mm_bid) / 2
                        else:
                            fair_value = current_mid_price
                else:
                    # Fallback to market microstructure or mid price
                    if mm_ask is not None and mm_bid is not None:
                        fair_value = (mm_ask + mm_bid) / 2
                    else:
                        fair_value = current_mid_price

            # Store current mid price for next iteration
            traderObject["squid_ink_last_price"] = current_mid_price
            
            # Adjust trading parameters based on volatility and z-score
            if volatility is not None:
                # In high volatility regimes, widen the take_width to be more conservative
                # In low volatility regimes, narrow the take_width to be more aggressive
                base_take_width = self.params[Product.SQUID_INK]["take_width"]
                volatility_adjusted_take_width = base_take_width * (1 + volatility * 5)
                
                # Further adjust based on z-score - be more aggressive when z-score is high
                if "squid_ink_zscore" in traderObject:
                    zscore = traderObject["squid_ink_zscore"]
                    zscore_adjustment = max(0, abs(zscore) - 1.5) * 0.2  # Only adjust when |z| > 1.5
                    if zscore > 0:  # Price is high, expect reversion down
                        # Be more aggressive in selling (narrow take_width for sells)
                        sell_take_width = volatility_adjusted_take_width * (1 - zscore_adjustment)
                        buy_take_width = volatility_adjusted_take_width * (1 + zscore_adjustment)
                    else:  # Price is low, expect reversion up
                        # Be more aggressive in buying (narrow take_width for buys)
                        buy_take_width = volatility_adjusted_take_width * (1 - zscore_adjustment)
                        sell_take_width = volatility_adjusted_take_width * (1 + zscore_adjustment)
                    
                    # Store asymmetric take_widths
                    traderObject["squid_ink_buy_take_width"] = buy_take_width
                    traderObject["squid_ink_sell_take_width"] = sell_take_width
                else:
                    # Store in trader object for use in take_orders
                    traderObject["squid_ink_volatility_adjusted_take_width"] = volatility_adjusted_take_width
                
                # Also adjust the clear_width based on volatility
                base_clear_width = self.params[Product.SQUID_INK]["clear_width"]
                volatility_adjusted_clear_width = base_clear_width * (1 + volatility * 3)
                traderObject["squid_ink_volatility_adjusted_clear_width"] = volatility_adjusted_clear_width
            
            return fair_value
        return None

    def kelp_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = self.get_swmid(order_depth)
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def calculate_basket_fair_value(self, basket: str, state: TradingState) -> float:
        """Calculate the fair value of a basket based on its components"""
        if basket not in BASKET_COMPOSITIONS:
            return None
            
        basket_value = 0
        for product, quantity in BASKET_COMPOSITIONS[basket].items():
            if product not in state.order_depths:
                return None
                
            order_depth = state.order_depths[product]
            if not order_depth.buy_orders or not order_depth.sell_orders:
                return None
                
            mid_price = self.get_swmid(order_depth)
            basket_value += mid_price * quantity
            
        return basket_value

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

        orders, buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
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
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
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
        disregard_edge: float,  # disregard trades within this edge for pennying or joining
        join_edge: float,  # join trades within this edge
        default_edge: float,  # default edge to request if there are no levels to penny or join
        manage_position: bool = False,
        soft_position_limit: int = 0,
        # will penny all other levels with higher edge
    ):
        orders: List[Order] = []
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair != None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_value - default_edge)
        if best_bid_below_fair != None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -1 * soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def execute_basket_arbitrage(self, basket_type: str, order_depths: Dict[str, OrderDepth], basket_position: int) -> Dict[str, List[Order]]:
        """Execute basket arbitrage when profitable"""
        result = {product: [] for product in BASKET_COMPOSITIONS[basket_type].keys()}
        result[basket_type] = []
        
        # Check if basket exists in order depths
        if basket_type not in order_depths:
            return result
            
        basket_depth = order_depths[basket_type]
        
        # Verify basket order depth has orders
        if not basket_depth.buy_orders or not basket_depth.sell_orders:
            return result
        
        # Get synthetic basket depth
        synthetic_depth = self.get_synthetic_basket_order_depth(order_depths, basket_type)
        
        # Verify synthetic depth has orders
        if not synthetic_depth.buy_orders or not synthetic_depth.sell_orders:
            return result
                
        basket_swmid = self.get_swmid(basket_depth)
        synthetic_swmid = self.get_swmid(synthetic_depth)
        
        # Return early if we can't calculate either SWMID
        if basket_swmid is None or synthetic_swmid is None:
            return result
                
        spread = basket_swmid - synthetic_swmid
        
        # Arbitrage thresholds from params
        min_spread = self.params[basket_type]["spread_threshold"]
        max_trade_size = self.params[basket_type]["max_trade_size"]
        
        if spread > min_spread:  # Sell basket, buy components
            try:
                basket_quantity = min(
                    abs(basket_depth.buy_orders[max(basket_depth.buy_orders.keys())]),
                    max_trade_size,
                    self.LIMIT[basket_type] + basket_position
                )
                
                if basket_quantity > 0:
                    # Sell basket
                    result[basket_type].append(Order(
                        basket_type,
                        max(basket_depth.buy_orders.keys()),
                        -basket_quantity
                    ))
                    
                    # Buy components
                    for product, quantity in BASKET_COMPOSITIONS[basket_type].items():
                        result[product].append(Order(
                            product,
                            min(order_depths[product].sell_orders.keys()),
                            quantity * basket_quantity
                        ))
                        
            except (KeyError, ValueError):
                return result
                    
        elif spread < -min_spread:  # Buy basket, sell components
            try:
                basket_quantity = min(
                    abs(basket_depth.sell_orders[min(basket_depth.sell_orders.keys())]),
                    max_trade_size,
                    self.LIMIT[basket_type] - basket_position
                )
                
                if basket_quantity > 0:
                    # Buy basket
                    result[basket_type].append(Order(
                        basket_type,
                        min(basket_depth.sell_orders.keys()),
                        basket_quantity
                    ))
                    
                    # Sell components
                    for product, quantity in BASKET_COMPOSITIONS[basket_type].items():
                        result[product].append(Order(
                            product,
                            max(order_depths[product].buy_orders.keys()),
                            -quantity * basket_quantity
                        ))
                        
            except (KeyError, ValueError):
                return result
    
        return result
    
    def get_synthetic_basket_order_depth(self, order_depths: Dict[str, OrderDepth], basket_type: str) -> OrderDepth:
        """Calculate synthetic order depth for a basket"""
        synthetic_depth = OrderDepth()
        
        # Get basket components and weights
        components = BASKET_COMPOSITIONS[basket_type]
        
        # Calculate best bids and asks for components
        best_prices = {}
        for product, quantity in components.items():
            if product not in order_depths or not order_depths[product].buy_orders or not order_depths[product].sell_orders:
                return synthetic_depth
                
            best_prices[product] = {
                'bid': max(order_depths[product].buy_orders.keys()),
                'ask': min(order_depths[product].sell_orders.keys()),
                'bid_vol': order_depths[product].buy_orders[max(order_depths[product].buy_orders.keys())],
                'ask_vol': abs(order_depths[product].sell_orders[min(order_depths[product].sell_orders.keys())])
            }
        
        # Calculate implied prices
        implied_bid = sum(best_prices[p]['bid'] * q for p, q in components.items())
        implied_ask = sum(best_prices[p]['ask'] * q for p, q in components.items())
        
        # Calculate volumes

        # taking the max is more risky but seems to yield better results
        implied_bid_vol = max(best_prices[p]['bid_vol'] // q for p, q in components.items())
        implied_ask_vol = max(best_prices[p]['ask_vol'] // q for p, q in components.items())
        
        if implied_bid_vol > 0:
            synthetic_depth.buy_orders[implied_bid] = implied_bid_vol
        if implied_ask_vol > 0:
            synthetic_depth.sell_orders[implied_ask] = -implied_ask_vol
            
        return synthetic_depth

    def get_swmid(self, order_depth: OrderDepth) -> float:
        """Calculate size-weighted mid price"""
        # Check if there are any orders
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
            
        try:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            best_bid_vol = abs(order_depth.buy_orders[best_bid])
            best_ask_vol = abs(order_depth.sell_orders[best_ask])
            
            return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
                best_bid_vol + best_ask_vol
            )
        except (ValueError, KeyError):
            return None

    def execute_basket_to_basket_arbitrage(self, state: TradingState) -> Dict[str, List[Order]]:
        """Execute arbitrage between PICNIC_BASKET1 and PICNIC_BASKET2"""
        result = {
            Product.PICNIC_BASKET1: [],
            Product.PICNIC_BASKET2: [],
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBES: []
        }
        
        # Check if both baskets exist in order depths
        if (Product.PICNIC_BASKET1 not in state.order_depths or 
            Product.PICNIC_BASKET2 not in state.order_depths):
            return result
            
        pb1_depth = state.order_depths[Product.PICNIC_BASKET1]
        pb2_depth = state.order_depths[Product.PICNIC_BASKET2]
        
        # Verify basket order depths have orders
        if (not pb1_depth.buy_orders or not pb1_depth.sell_orders or
            not pb2_depth.buy_orders or not pb2_depth.sell_orders):
            return result
        
        # Calculate the theoretical relationship between the baskets
        # PB1 = 6 CROISSANTS + 3 JAMS + 1 DJEMBE
        # PB2 = 4 CROISSANTS + 2 JAMS
        # Therefore, 3 PB2 + 3 DJEMBE = 2 PB1 (ignoring transaction costs)
        
        # Get best bid/ask for each basket
        pb1_best_bid = max(pb1_depth.buy_orders.keys())
        pb1_best_ask = min(pb1_depth.sell_orders.keys())
        pb2_best_bid = max(pb2_depth.buy_orders.keys())
        pb2_best_ask = min(pb2_depth.sell_orders.keys())
        
        # Get positions
        pb1_position = state.position.get(Product.PICNIC_BASKET1, 0)
        pb2_position = state.position.get(Product.PICNIC_BASKET2, 0)
        djembe_position = state.position.get(Product.DJEMBES, 0)
        
        # Check if DJEMBE is available for trading
        if Product.DJEMBES not in state.order_depths:
            return result
            
        djembe_depth = state.order_depths[Product.DJEMBES]
        if not djembe_depth.buy_orders or not djembe_depth.sell_orders:
            return result
            
        djembe_best_bid = max(djembe_depth.buy_orders.keys())
        djembe_best_ask = min(djembe_depth.sell_orders.keys())
        
        # Define arbitrage thresholds
        arb_threshold = 5  # Minimum profit required to execute arbitrage
        max_trade_size = 10  # Maximum trade size for arbitrage
        
        # Arbitrage opportunity 1: 
        # Buy 2 PB1, sell 3 PB2 + 3 DJEMBE
        spread1 = (3 * pb2_best_bid + 3 * djembe_best_bid) - (2 * pb1_best_ask)
        
        if spread1 > arb_threshold:
            # Calculate trade size
            max_pb1_buy = min(abs(pb1_depth.sell_orders[pb1_best_ask]), 
                             (self.LIMIT[Product.PICNIC_BASKET1] - pb1_position) // 2)
            max_pb2_sell = min(abs(pb2_depth.buy_orders[pb2_best_bid]), 
                              (self.LIMIT[Product.PICNIC_BASKET2] + pb2_position) // 3)
            max_djembe_sell = min(abs(djembe_depth.buy_orders[djembe_best_bid]), 
                                (self.LIMIT[Product.DJEMBES] + djembe_position) // 3)
            
            trade_size = min(max_pb1_buy, max_pb2_sell, max_djembe_sell, max_trade_size)
            
            if trade_size > 0:
                # Buy 2*trade_size PB1
                result[Product.PICNIC_BASKET1].append(Order(
                    Product.PICNIC_BASKET1, 
                    pb1_best_ask,
                    2 * trade_size
                ))
                
                # Sell 3*trade_size PB2
                result[Product.PICNIC_BASKET2].append(Order(
                    Product.PICNIC_BASKET2,
                    pb2_best_bid,
                    -3 * trade_size
                ))
                
                # Sell 3*trade_size DJEMBE
                result[Product.DJEMBES].append(Order(
                    Product.DJEMBES,
                    djembe_best_bid,
                    -3 * trade_size
                ))
        
        # Arbitrage opportunity 2:
        # Buy 3 PB2 + 3 DJEMBE, sell 2 PB1
        spread2 = (2 * pb1_best_bid) - (3 * pb2_best_ask + 3 * djembe_best_ask)
        
        if spread2 > arb_threshold:
            # Calculate trade size
            max_pb1_sell = min(abs(pb1_depth.buy_orders[pb1_best_bid]), 
                              (self.LIMIT[Product.PICNIC_BASKET1] + pb1_position) // 2)
            max_pb2_buy = min(abs(pb2_depth.sell_orders[pb2_best_ask]), 
                             (self.LIMIT[Product.PICNIC_BASKET2] - pb2_position) // 3)
            max_djembe_buy = min(abs(djembe_depth.sell_orders[djembe_best_ask]), 
                               (self.LIMIT[Product.DJEMBES] - djembe_position) // 3)
            
            trade_size = min(max_pb1_sell, max_pb2_buy, max_djembe_buy, max_trade_size)
            
            if trade_size > 0:
                # Sell 2*trade_size PB1
                result[Product.PICNIC_BASKET1].append(Order(
                    Product.PICNIC_BASKET1,
                    pb1_best_bid,
                    -2 * trade_size
                ))
                
                # Buy 3*trade_size PB2
                result[Product.PICNIC_BASKET2].append(Order(
                    Product.PICNIC_BASKET2,
                    pb2_best_ask,
                    3 * trade_size
                ))
                
                # Buy 3*trade_size DJEMBE
                result[Product.DJEMBES].append(Order(
                    Product.DJEMBES,
                    djembe_best_ask,
                    3 * trade_size
                ))
        
        return result

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFOREST_RESIN in self.params and Product.RAINFOREST_RESIN in state.order_depths:
            resin_position = (
                state.position[Product.RAINFOREST_RESIN]
                if Product.RAINFOREST_RESIN in state.position
                else 0
            )
            resin_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["take_width"],
                    resin_position,
                )
            )
            resin_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFOREST_RESIN,
                    state.order_depths[Product.RAINFOREST_RESIN],
                    self.params[Product.RAINFOREST_RESIN]["fair_value"],
                    self.params[Product.RAINFOREST_RESIN]["clear_width"],
                    resin_position,
                    buy_order_volume,
                    sell_order_volume,
                )
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
            result[Product.RAINFOREST_RESIN] = (
                resin_take_orders + resin_clear_orders + resin_make_orders
            )

        if Product.SQUID_INK in self.params and Product.SQUID_INK in state.order_depths:
            squid_ink_position = (
                state.position[Product.SQUID_INK]
                if Product.SQUID_INK in state.position
                else 0
            )
            # Calculate fair value using GARCH model
            squid_ink_fair_value = self.squid_ink_fair_value(
                state.order_depths[Product.SQUID_INK], traderObject
            )
            
            # Extract z-score if available
            zscore = traderObject.get("squid_ink_zscore", 0)
            
            # Use volatility-adjusted parameters if available
            take_width = (
                traderObject.get("squid_ink_volatility_adjusted_take_width",
                self.params[Product.SQUID_INK]["take_width"])
            )
            clear_width = (
                traderObject.get("squid_ink_volatility_adjusted_clear_width",
                self.params[Product.SQUID_INK]["clear_width"])
            )
            
            # Get asymmetric take widths if available
            buy_take_width = traderObject.get("squid_ink_buy_take_width", None)
            sell_take_width = traderObject.get("squid_ink_sell_take_width", None)
            
            # Get current volatility for position sizing
            volatility = traderObject.get("squid_ink_volatility", 0.01)
            
            # Adjust position limit based on volatility and z-score
            # Take smaller positions in high volatility, but larger when expecting mean reversion
            volatility_factor = 1 / (1 + 5 * volatility)
            zscore_factor = 1 + (max(0, abs(zscore) - 2) * 0.1)  # Increase position size when |z| > 2
            effective_position_limit = int(self.LIMIT[Product.SQUID_INK] * volatility_factor * zscore_factor)
            
            # Take orders with volatility-adjusted and asymmetric parameters
            buy_order_volume = 0
            sell_order_volume = 0
            squid_ink_orders = []
            
            squid_ink_take_orders, buy_order_volume, sell_order_volume = (
                self.take_best_orders(
                    Product.SQUID_INK,
                    squid_ink_fair_value,
                    take_width,
                    orders=squid_ink_orders.copy(),
                    order_depth=state.order_depths[Product.SQUID_INK],
                    position=squid_ink_position,
                    buy_order_volume=buy_order_volume,
                    sell_order_volume=sell_order_volume,
                    prevent_adverse=self.params[Product.SQUID_INK]["prevent_adverse"],
                    adverse_volume=self.params[Product.SQUID_INK]["adverse_volume"],
                    buy_take_width=buy_take_width,
                    sell_take_width=sell_take_width,
                )
            )
            
            # Clear orders with volatility-adjusted parameters
            squid_ink_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.SQUID_INK,
                    state.order_depths[Product.SQUID_INK],
                    squid_ink_fair_value,
                    clear_width,
                    squid_ink_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            
            # Make orders - adjust edge based on volatility and z-score
            # In high volatility, widen the spread to reduce risk
            edge_adjustment = 1 + 10 * volatility
            
            # Further adjust based on z-score
            if abs(zscore) > 2:
                # When expecting mean reversion, tighten spread on the side we expect to be profitable
                if zscore > 0:  # Price is high, expecting a drop
                    # Make more aggressive sell offers (smaller edge)
                    adjusted_default_edge = self.params[Product.SQUID_INK]["default_edge"] * edge_adjustment * 0.8
                else:  # Price is low, expecting a rise
                    # Make more aggressive buy offers (smaller edge)
                    adjusted_default_edge = self.params[Product.SQUID_INK]["default_edge"] * edge_adjustment * 0.8
            else:
                adjusted_default_edge = self.params[Product.SQUID_INK]["default_edge"] * edge_adjustment
            
            squid_ink_make_orders, _, _ = self.make_orders(
                Product.SQUID_INK,
                state.order_depths[Product.SQUID_INK],
                squid_ink_fair_value,
                squid_ink_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUID_INK]["disregard_edge"],
                self.params[Product.SQUID_INK]["join_edge"],
                adjusted_default_edge,
            )
            
            result[Product.SQUID_INK] = (
                squid_ink_take_orders + squid_ink_clear_orders + squid_ink_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            kelp_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            kelp_fair_value = self.kelp_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            kelp_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["take_width"],
                    kelp_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            kelp_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    kelp_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    kelp_position,
                    buy_order_volume,
                    sell_order_volume,
                )
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
            result[Product.KELP] = (
                kelp_take_orders + kelp_clear_orders + kelp_make_orders
            )

        # Execute basket-to-basket arbitrage first (priority)
        basket_to_basket_orders = self.execute_basket_to_basket_arbitrage(state)
        
        # Add basket-to-basket arbitrage orders to result
        for product, orders in basket_to_basket_orders.items():
            if orders:
                if product not in result:
                    result[product] = []
                result[product].extend(orders)

        # Get products that already have arbitrage orders
        products_with_arb_orders = {product for product, orders in basket_to_basket_orders.items() if orders}

        # Process individual components only if they don't have arbitrage orders
        for product in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]:
            if product in state.order_depths and product not in products_with_arb_orders:
                position = state.position.get(product, 0)
                
                # Calculate fair value as mid price
                order_depth = state.order_depths[product]
                if order_depth.buy_orders and order_depth.sell_orders:
                    fair_value = self.get_swmid(order_depth)
                    
                    # Take orders
                    product_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                        product,
                        order_depth,
                        fair_value,
                        self.params[product]["take_width"],
                        position,
                        self.params[product]["prevent_adverse"],
                        self.params[product]["adverse_volume"],
                    )
                    
                    # Clear orders
                    product_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                        product,
                        order_depth,
                        fair_value,
                        self.params[product]["clear_width"],
                        position,
                        buy_order_volume,
                        sell_order_volume,
                    )
                    
                    # Make orders
                    product_make_orders, _, _ = self.make_orders(
                        product,
                        order_depth,
                        fair_value,
                        position,
                        buy_order_volume,
                        sell_order_volume,
                        self.params[product]["disregard_edge"],
                        self.params[product]["join_edge"],
                        self.params[product]["default_edge"],
                    )
                    
                    if product not in result:
                        result[product] = []
                    result[product].extend(product_take_orders + product_clear_orders + product_make_orders)

        # Regular basket arbitrage for baskets not involved in basket-to-basket arbitrage
        for basket in [Product.PICNIC_BASKET1, Product.PICNIC_BASKET2]:
            if basket in state.order_depths and basket not in products_with_arb_orders:
                basket_position = state.position.get(basket, 0)
                
                # Execute basket arbitrage
                basket_orders = self.execute_basket_arbitrage(
                    basket,
                    state.order_depths,
                    basket_position
                )
                
                # Add orders to result
                for product, orders in basket_orders.items():
                    if orders and product not in products_with_arb_orders:
                        if product not in result:
                            result[product] = []
                        result[product].extend(orders)

        conversions = 1
        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData
