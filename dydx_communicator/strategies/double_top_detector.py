import logging
import pandas as pd
from typing import Tuple, Union, Optional

class DoubleTopDetector:

    """
    A class to detect double top patterns in financial time series data.

    Attributes:
        current_row (pd.series): Last row [-1] in a pandas DataFrame with OHLCV data.
        n_periods_to_confirm_swing (int): The number of consecutive periods where High < ressist zone lower to confirm a swing.
        invalidation_n (int): The maximum number of periods allowed after the swing is detected before invalidating the setup.
        ressist_zone_upper (float, None): The upper limit of the ressist zone.
        ressist_zone_lower (float, None): The lower limit of the ressist zone.

    Methods:
        setup_logger() -> logging.Logger:
            Sets up a logger for the DoubleTopDetector class, with both file and console handlers.
        
        __init__(n_periods_to_confirm_swing: int, invalidation_n: int) -> None:
            Initializes the DoubleBottomDetector with the given parameters.

        reset() -> None:
            Resets the state of the detector to start a new detection cycle.
        
        detect() -> Union[None, pd.Timestamp]:
            Analyzes the current_row of time series data to detect a double top pattern.
            If a double top pattern is detected, returns the timestamp of the last row.
            Otherwise, returns None.

        Example:

            detector = DoubleTopDetector(n_periods_to_confirm_swing=5, invalidation_n=72)
            
            detector.ressist_zone_upper = float(1000)
            detector.resssit_zone_lower = float(800)
            detector.current_row = ohlcv[-1]
            res = detector.detect()

            "res": timestamp of the triggercandle.
        """

    @staticmethod
    def setup_logger() -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        fh = logging.FileHandler('resistance_zone_detector.log')
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    logger = setup_logger()


    def __init__(self, n_periods_to_confirm_swing: int, invalidation_n: int) -> None:

        self.n_periods_to_confirm_swing = n_periods_to_confirm_swing
        self.invalidation_n = invalidation_n

        self.resistance_zone_upper: Union[float, None] = None
        self.resistance_zone_lower: Union[float, None] = None

        self.current_row: pd.Series = pd.Series(dtype='float64')

        self.timestamp_for_first_touch: Union[pd.Timestamp, None] = None
        self.sell_zone: Union[Tuple[float, float], None] = None
        self.invalidation_cnt: int = 0
        
        self.swing_detected: bool = False

        self.invalidation_period_cnt: int = 0
        self.candle_counter: int = 0


    def reset(self) -> None:
        self.timestamp_for_first_touch = None
        self.candle_counter = 0
        self.invalidation_cnt = 0
        self.swing_detected = False
        self.logger.info('Detection Cycle ended')


    def detect(self): # -> Tuple[Optional[pd.Series], Optional[str]]:

        if self.timestamp_for_first_touch is None:
   
            if self.current_row.Open < self.resistance_zone_lower and self.current_row.High >= self.resistance_zone_lower and self.current_row.Close <= self.resistance_zone_upper:
                print('HERE')
                self.timestamp_for_first_touch = self.current_row.Index
                self.sell_zone = (max(self.current_row.Close, self.resistance_zone_lower), self.current_row.High)

                self.logger.info(f'Resistance Zone upper: {self.resistance_zone_upper}')
                self.logger.info(f'Resistance Zone Lower: {self.resistance_zone_lower}')
                self.logger.info(f'First touch of resistance zone at: {self.timestamp_for_first_touch} - High:{self.current_row.High}')
                self.logger.info(f"Sell zone: {self.sell_zone}")
        
        else:

            if self.current_row.Close > self.resistance_zone_upper:
                
                self.logger.warning(f"Candle closed above upper resistance zone at: {self.current_row.Index}. Setup invalid.")
                self.reset()
                return None
            
            elif self.current_row.Close > self.sell_zone[1]:

                self.logger.warning(f"Candle closed above sell zone at: {self.current_row.Index}. Setup invalid.")
                self.reset()
                return None
            
            if self.swing_detected:
                self.invalidation_cnt += 1

                if self.invalidation_cnt > self.invalidation_n:
                
                    self.logger.warning(f"Double top not forfilled within {self.invalidation_n} periods. Setup invalid.")
                    self.reset()
                    return None
            
                elif self.current_row.High >= self.sell_zone[0]:
                    self.logger.info(f"Price in sell zone: {self.current_row.Index}")
                    self.reset()
                    return (self.current_row, "SELL")

            else:

                if self.current_row.High < self.resistance_zone_lower:
                
                    self.candle_counter += 1

                    if self.candle_counter == self.n_periods_to_confirm_swing:

                        self.logger.info(f'Swing after first touch detected: {self.current_row.Index}')
                        self.swing_detected = True

                elif self.current_row.High > self.resistance_zone_lower:

                    self.candle_counter = 0
