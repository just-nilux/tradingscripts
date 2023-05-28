import datetime
import logging

logger = logging.getLogger(__name__)

class StrategyExecutor:
    def __init__(self, client, detectors, atrs, liq_levels, first_iteration, symbol, timeframe, df):
        self.client = client
        self.detectors = detectors
        self.atrs = atrs
        self.liq_levels = liq_levels
        self.first_iteration = first_iteration
        self.symbol = symbol
        self.timeframe = timeframe
        self.df = df


    @staticmethod
    def timeframe_to_minutes(timeframe):
        units = {
       "MIN": 1,
       "MINS": 1,
       "HOUR": 60,
       "HOURS": 60,
       "DAY": 1440,
       }

        # Split the input string into numbers and text parts
        timeframe_value = int(''.join(filter(str.isdigit, timeframe)))
        timeframe_unit = ''.join(filter(str.isalpha, timeframe))

        return timeframe_value * units[timeframe_unit]

    def fetch_support_resistance(self):

        # fetch support and resistance levels
        liq_data = self.liq_levels[self.symbol]  # Retrieve the list of values for 'symbol'

        # Assign the values to variables
        support_lower = min(liq_data)
        support_upper = sorted(liq_data)[1]  # Second lowest value

        resistance_upper = max(liq_data)
        resistance_lower = sorted(liq_data)[-2]  # Second highest value


        return support_upper, support_lower, resistance_upper, resistance_lower



    def execute_strategies(self):
        current_time = datetime.datetime.now()
        minutes = current_time.minute
        timeframe_minutes = self.timeframe_to_minutes(self.timeframe)

        if not (minutes % timeframe_minutes):
            if self.first_iteration:
                self.add_all_candles_to_atr()
            else:
                self.add_last_candle_to_atr()

            support_upper, support_lower, resistance_upper, resistance_lower = self.fetch_support_resistance()

            for strategy_function_name in self.client.config['strategies'][0]['strategy_functions']:
                self.execute_strategy_and_handle_signal(strategy_function_name, support_upper, support_lower, resistance_upper, resistance_lower)



    def add_all_candles_to_atr(self):
        for _, candle in self.df.iterrows():
            atr = self.atrs[f"{self.symbol}_{self.timeframe}"]
            atr.add_input_value(candle)
            if not atr:
                logger.info(f"ATR not available for {self.symbol} on {self.timeframe} - no. input values: {len(atr.input_values)} - Needs: {atr.period}")
        self.last_closed_candle = self.df.iloc[-1]



    def add_last_candle_to_atr(self):
        self.last_closed_candle = self.df.iloc[-1]
        atr = self.atrs[f"{self.symbol}_{self.timeframe}"]
        atr.add_input_value(self.last_closed_candle)



    def execute_strategy_and_handle_signal(self, strategy_function_name, support_upper, support_lower, resistance_upper, resistance_lower):
        try:
            signal = self.execute_strategy(strategy_function_name, support_upper, support_lower, resistance_upper, resistance_lower)
            self.handle_signal(signal, strategy_function_name)
        except Exception as e:
            logger.error(f"Error while executing strategy or signal for {self.symbol} on {self.timeframe}: {e}")



    def execute_strategy(self, strategy_function_name, support_upper, support_lower, resistance_upper, resistance_lower):
        logger.debug(f"Executing strategy {strategy_function_name} for {self.symbol} on {self.timeframe}")
        strategy_function = globals()[strategy_function_name]
        detector_key = f"{self.symbol}_{self.timeframe}_{strategy_function_name}"
        detector = self.detectors[detector_key]

        if strategy_function_name == "doubleBottomEntry":
            return strategy_function(self.last_closed_candle, detector, support_zone_upper=support_upper, support_zone_lower=support_lower)

        elif strategy_function_name == "doubleTopEntry":
            return strategy_function(self.last_closed_candle, detector, ressist_zone_upper=resistance_upper, ressist_zone_lower=resistance_lower)
        
        elif strategy_function_name == "liqSweepEntry":
            return strategy_function(self.last_closed_candle, detector, upper_liq_level=resistance_upper, lower_liq_level=support_lower )



    def handle_signal(self, signal, strategy_function_name):
        
        try:
            if signal[1] in ('SELL', 'BUY'):
                logger.debug(f"Executing signal {strategy_function_name} for {self.symbol} on {self.timeframe} - side: {signal[1]}")
                
                size = float(self.client.order_size(self.symbol, self.client.config['position_size']))
                logger.debug(f"Placing {signal[1].lower()} order for {self.symbol} with size {size}")
                
                atr = self.atrs[f"{self.symbol}_{self.timeframe}"]
                order = self.client.place_market_order(symbol=self.symbol, size=size, side=signal[1], atr=atr[-1], trigger_candle=signal[0])

            elif signal[1] is None:
                logger.info(f"No signal for symbol: {self.symbol} on TF: {self.timeframe} - {strategy_function_name}")
            
            else:
                logger.warning(f"Invalid signal: {signal[1]}")
        
        except Exception as e:
            logger.error(f"Error while executing signal for {self.symbol} on {self.timeframe}: {e} - side: {signal[1]}")
