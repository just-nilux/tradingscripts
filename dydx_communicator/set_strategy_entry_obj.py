from logger_setup import setup_logger

logger = setup_logger(__name__)




def doubleTopEntry(last_closed_candle, detector, ressist_zone_upper, ressist_zone_lower):

    logger.debug(f"Executing doubleTopEntry for detector {detector}")

    detector.resistance_zone_upper = ressist_zone_upper
    detector.resistance_zone_lower = ressist_zone_lower
    detector.current_row = last_closed_candle
    res = detector.detect()

    if isinstance(res, tuple) and res[1] == 'SELL':
        return res
    return (None, None)



def doubleBottomEntry(last_closed_candle, detector, support_zone_upper, support_zone_lower):

    logger.debug(f"Executing doubleBottomEntry for detector {detector}")

    detector.support_zone_upper = support_zone_upper
    detector.support_zone_lower = support_zone_lower
    detector.current_row = last_closed_candle
    res = detector.detect()

    if isinstance(res, tuple) and res[1] == 'BUY':
        return res
    return (None, None)


def liqSweepEntry(last_closed_candle, detector, upper_liq_level, lower_liq_level):

    logger.debug(f"Executing LiqSweepEntry strategy for {detector}")

    detector.upper_liq_level = upper_liq_level
    detector.lower_liq_level = lower_liq_level
    detector.current_row = last_closed_candle
    res = detector.detect()

    if isinstance(res, tuple) and res[1] in ('BUY', 'SELL'):
        return res
    return (None, None)

