def order_size(self, symbol: str, position_size: float) -> Optional[float]:
    """
    Calculate the order size based on the available equity, leverage, and market data.

    Parameters:
    symbol (str): The symbol of the asset for which the order size is to be calculated.
    position_size (float): The proportion of free equity to use for the order.

    Returns:
    float: The calculated order size.

    This function first retrieves the leverage for the specified symbol and calculates the 
    amount of free equity that should be used for the order. It then fetches the current 
    market data for the symbol, including the oracle price, the minimum order size, and the step size.

    The function then calculates the order size based on the free equity, the leverage, and 
    the oracle price, ensuring that the order size is a multiple of the step size. 
    It returns this calculated order size.
    """
    try:
        leverage = self.config['leverage']
        if leverage is None:
            raise ValueError(f"Leverage not found for symbol: {symbol}")

        if not 0 < position_size <= 1:
            raise ValueError("position_size should be a value between 0 and 1")

        free_equity = self.fetch_free_equity() * position_size
        market_data = self.client.public.get_markets(market=symbol).data['markets'][symbol]
        if market_data is None:
            raise ValueError(f"Market data not found for symbol: {symbol}")

        price = float(market_data['oraclePrice'])
        if price <= 0:
            raise ValueError(f"Invalid oracle price: {price}")

        min_order_size = float(market_data['minOrderSize'])
        if min_order_size <= 0:
            raise ValueError(f"Invalid minimum order size: {min_order_size}")

        step_size = float(market_data['stepSize'])
        if step_size <= 0:
            raise ValueError(f"Invalid step size: {step_size}")

        # Ensure the order size is divisible by the step size
        raw_order_size = (free_equity / price) * leverage
        calculated_order_size = step_size * round(raw_order_size / step_size)

        if calculated_order_size < min_order_size:
            self.logger.error(f"Calculated order size for {symbol} is less than the minimum order size")
            return None

        self.logger.info(f"Calculated order size for {symbol} is {calculated_order_size}")
        return calculated_order_size

    except Exception as e:
        self.logger.error(f"Error calculating order size for {symbol}: {e}")
        return None
