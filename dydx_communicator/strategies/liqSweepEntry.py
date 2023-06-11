import logging
import pandas as pd
from typing import Tuple, Union, Optional
from logger_setup import setup_logger


class SweepDetector:


    def __init__(self, n_periods_to_confirm_sweep: int, cross_pct_threshold: float) -> None:

        self.upper_liq_level: Union[float, None] = None
        self.lower_liq_level: Union[float, None] = None

        self.n_periods_to_confirm_sweep: int = n_periods_to_confirm_sweep
        self.cross_pct_threshold: float = cross_pct_threshold

        self.current_row: pd.Series = pd.Series(dtype='float64')

        self.candle_for_first_cross_of_liq_level: Union[pd.Series, None] = None

        self.invalidation_cnt: int = 0

        self.cross_of_upper_liq: bool = False
        self.cross_of_lower_liq: bool = False
        self.sweep_crossed_with_min_req_pct: bool = False

        self.logger = setup_logger(__name__)


    def reset(self) -> None:
        self.cross_of_upper_liq = False
        self.cross_of_lower_liq = False
        self.sweep_crossed_with_min_req_pct = False
        self.candle_for_first_cross_of_liq_level = None
        self.invalidation_cnt = 0
        self.logger.info('Detection Cycle ended')

    
    def detect_sweep_magnitude(self, side: str):
        
        if side == "upside_liq":
            price_above_liq_level_pct = ((self.current_row.high - self.upper_liq_level) / self.upper_liq_level) * 100
            if price_above_liq_level_pct >= self.cross_pct_threshold:
                self.logger.info(f'Price {self.current_row.high} crossed UPPER liq. level {self.upper_liq_level} with a threshold of: {round(price_above_liq_level_pct,2)}%')
                self.sweep_crossed_with_min_req_pct = True


        elif side == "downside_liq":
            price_below_liq_level_pct = ((self.lower_liq_level - self.current_row.low) / self.lower_liq_level) * 100
            if price_below_liq_level_pct >= self.cross_pct_threshold:
                self.logger.info(f'Price {self.current_row.low} crossed LOWER liq. level {self.lower_liq_level} with: {round(price_below_liq_level_pct,2)}%')
                self.sweep_crossed_with_min_req_pct = True

            



    def detect(self):
        

        if self.candle_for_first_cross_of_liq_level is None:

            # For upper liquidation level
            if self.current_row.low < self.upper_liq_level < self.current_row.close:
                self.logger.info(f'Upper liq. level: {self.upper_liq_level}')
                self.logger.info(f'Lower liq. level: {self.lower_liq_level}')

                self.candle_for_first_cross_of_liq_level = self.current_row
                self.logger.info(f'First cross of UPPER liq. level at: {self.candle_for_first_cross_of_liq_level.name} - close:{self.current_row.close}')
                self.cross_of_upper_liq = True
                self.detect_sweep_magnitude("upside_liq")

            # For lower liquidation level
            elif self.current_row.high > self.lower_liq_level > self.current_row.close:
                self.logger.info(f'Upper liq. level: {self.upper_liq_level}')
                self.logger.info(f'Lower liq. level: {self.lower_liq_level}')

                self.candle_for_first_cross_of_liq_level = self.current_row
                self.logger.info(f'First cross of LOWER liq. level at: {self.candle_for_first_cross_of_liq_level.name} - close:{self.current_row.close}')
                self.cross_of_lower_liq = True
                self.detect_sweep_magnitude("downside_liq")
            else:
                return

        if self.cross_of_upper_liq:

            self.invalidation_cnt +=1


            if not self.sweep_crossed_with_min_req_pct:
                self.detect_sweep_magnitude("upside_liq")


            if self.invalidation_cnt > self.n_periods_to_confirm_sweep:
                self.logger.warning(f"Sweep did not happen within {self.n_periods_to_confirm_sweep} periods. Setup invalid.")
                self.reset()
                return None

            elif self.current_row.close < self.upper_liq_level and not self.sweep_crossed_with_min_req_pct:
                self.logger.warning(f"Sweep did not cross UPPER liq. level with the minimum requirement {self.cross_pct_threshold}%")
                self.reset()
                return None

            elif self.current_row.close < self.upper_liq_level and self.sweep_crossed_with_min_req_pct and self.invalidation_cnt <= self.n_periods_to_confirm_sweep:
                self.logger.info(f'UPPER liq. sweep fulfilled at {self.current_row.name} - close: {self.current_row.close}')
                self.reset()
                return (self.current_row, "SELL", "Liq_Sweep_Entry")
                    


        elif self.cross_of_lower_liq:

            self.invalidation_cnt +=1

            if not self.sweep_crossed_with_min_req_pct:
                self.detect_sweep_magnitude("downside_liq")


            if self.invalidation_cnt > self.n_periods_to_confirm_sweep:
                self.logger.warning(f"Sweep did not happen within {self.n_periods_to_confirm_sweep} periods. Setup invalid.")
                self.reset()
                return 'invalid' #None
            
            elif self.current_row.close > self.lower_liq_level and not self.sweep_crossed_with_min_req_pct:
                self.logger.warning(f"Sweep did not cross LOWER liq. level with the minimum requirement {self.cross_pct_threshold}%")
                self.reset()
                return 'invalid' #None

            
            elif self.current_row.close > self.lower_liq_level and self.sweep_crossed_with_min_req_pct and self.invalidation_cnt <= self.n_periods_to_confirm_sweep:
                self.logger.info(f'LOWER liq. sweep fulfilled at {self.current_row.name} - close: {self.current_row.close}')
                self.reset()
                return (self.current_row, "BUY", "Liq_Sweep_Entry")
