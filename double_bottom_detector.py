import logging
import pandas as pd
from typing import List, Tuple, Union

class DoubleBottomDetector:

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
        self.first_touch_index: Union[int, None] = None
        
        self.swing_detected: bool = False

        self.invalidation_period_cnt: int = 0
        self.candle_counter: int = 0

    def reset(self) -> None:
        self.first_touch_index = None
        self.timestamp_for_first_touch = None
        self.candle_counter = 0
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

            self.first_touch_index = self.i
        
        elif self.timestamp_for_first_touch is not None:

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
                if self.candle_counter > self.n_periods_to_confirm_swing:

                    self.logger.info(f'Swing after first touch detected: {self.last_row_timestamp}')
                    self.swing_detected = True
            
            elif self.swing_detected and  self.i - self.first_touch_index > self.invalidation_n:
                
                self.logger.warning(f"Double bottom not forfilled within {self.invalidation_n} periods. Setup invalid.")
                self.reset()
                return self.last_row_timestamp
            
            elif self.swing_detected and self.last_row.Low <= self.buy_zone[0]:
                self.logger.info(f"Price in buy zone: {self.last_row_timestamp}")
                self.reset()
                return self.last_row_timestamp





