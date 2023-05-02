import logging
import pandas as pd
from typing import List, Tuple, Union



class DoubleBottomDetector:

    """
    A class to detect double bottom patterns in financial time series data.

    Attributes:
        last_row (pd.series): Last row [-1] in a pandas DataFrame with OHLCV data.
        n_periods_to_confirm_swing (int): The number of consecutive periods where Low > support zone upper to confirm a swing.
        invalidation_n (int): The maximum number of periods allowed after the swing is detected before invalidating the setup.
        support_zone_upper (float, None): The upper limit of the support zone.
        support_zone_lower (float, None): The lower limit of the support zone.

    Methods:
        setup_logger() -> logging.Logger:
            Sets up a logger for the DoubleBottomDetector class, with both file and console handlers.
        
        __init__(n_periods_to_confirm_swing: int, invalidation_n: int) -> None:
            Initializes the DoubleBottomDetector with the given parameters.

        reset() -> None:
            Resets the state of the detector to start a new detection cycle.
        
        detect() -> Union[None, pd.Timestamp]:
            Analyzes the last_row of time series data to detect a double bottom pattern.
            If a double bottom pattern is detected, returns the timestamp of the last row.
            Otherwise, returns None.

        Example:

            detector.support_zone_upper = float(1000)
            detector.support_zone_lower = float(800)
            detector.last_row = ohlcv[-1]
            detector.last_row_timestamp = ohlcv.Index
            res = detector.detect()
    """

    @staticmethod
    def setup_logger() -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # create file handler which logs even debug messages
        fh = logging.FileHandler('support_zone_detector.log')
        fh.setLevel(logging.DEBUG)

        # create console handler with a lower log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger
    


    logger = setup_logger()



    def __init__(self, n_periods_to_confirm_swing: int, invalidation_n: int) -> None:

        self.n_periods_to_confirm_swing = n_periods_to_confirm_swing
        self.invalidation_n = invalidation_n

        self.support_zone_upper: Union[float, None] = None
        self.support_zone_lower: Union[float, None] = None

        self.i: int = 0
        self.last_row: pd.Series = pd.Series(dtype='float64')

        self.last_row_timestamp: Union[pd.Timestamp, None] = None
        self.timestamp_for_first_touch: Union[pd.Timestamp, None] = None
        self.buy_zone: Union[Tuple[float, float], None] = None
        self.invalidation_cnt: int = 0
        
        self.swing_detected: bool = False

        self.invalidation_period_cnt: int = 0
        self.candle_counter: int = 0



    def reset(self) -> None:
        self.i = 0
        self.timestamp_for_first_touch = None
        self.candle_counter = 0
        self.invalidation_cnt = 0
        self.swing_detected = False
        self.logger.info('Detection Cycle ended')



    def detect(self) -> Union[None, pd.Timestamp]:

        if (self.last_row.Open > self.support_zone_upper and self.last_row.Low <= self.support_zone_upper and
            self.last_row.Close >= self.support_zone_lower and self.timestamp_for_first_touch is None):

            self.timestamp_for_first_touch = self.last_row_timestamp
            self.buy_zone = (min(self.last_row.Close, self.support_zone_upper), self.last_row.Low)

            self.logger.info(f'Support Zone upper: {self.support_zone_upper}')
            self.logger.info(f'Support Zone Lower: {self.support_zone_lower}')
            self.logger.info(f'First touch of support zone at: {self.timestamp_for_first_touch} - Low:{self.last_row.Low}')
            self.logger.info(f"Buy zone: {self.buy_zone}")
        
        elif self.timestamp_for_first_touch is not None:
            
            if self.swing_detected:
                self.invalidation_cnt += 1

            if self.last_row.Close < self.support_zone_lower:
                
                self.logger.warning(f"Candle closed below lower support zone at: {self.last_row_timestamp}. Setup invalid.")
                self.reset()
                return self.last_row_timestamp
            
            elif self.last_row.Close < self.buy_zone[1]:

                self.logger.warning(f"Candle closed below buy zone at: {self.last_row_timestamp}. Setup invalid.")
                self.reset()
                return self.last_row_timestamp
            
            elif self.last_row.Low > self.support_zone_upper and not self.swing_detected:

                self.candle_counter += 1

                if self.candle_counter == self.n_periods_to_confirm_swing:

                    self.logger.info(f'Swing after first touch detected: {self.last_row_timestamp}')
                    self.swing_detected = True

            elif self.last_row.Low < self.support_zone_upper and not self.swing_detected:

                self.candle_counter = 0
            
            elif self.swing_detected and self.invalidation_cnt > self.invalidation_n:
                
                self.logger.warning(f"Double bottom not forfilled within {self.invalidation_n} periods. Setup invalid.")
                self.reset()
                return self.last_row_timestamp
            
            elif self.swing_detected and self.last_row.Low <= self.buy_zone[0]:
                self.logger.info(f"Price in buy zone: {self.last_row_timestamp}")
                self.reset()
                return self.last_row_timestamp





