from strategies.doubleBottomEntry import DoubleBottomDetector
from strategies.doubleTopEntry import DoubleTopDetector
from strategies.liqSweepEntry import SweepDetector
from logger_setup import setup_logger
from talipp.indicators import ATR


logger = setup_logger(__name__)


def initialize_detectors(client: object, detectors=None, atrs=None):
    try:
        if detectors is None:
            detectors = {}

        if atrs is None:
            atrs = {}

        for strategy in client.config['strategies']:
            try:
                symbols = set(strategy['symbols'])
                timeframes = set(strategy['timeframes'])
                strategy_functions = strategy['strategy_functions']
            except KeyError as e:
                logger.error(f"Error while retrieving strategy details: Missing key {e}")
                continue

            for symbol in symbols:
                for timeframe in timeframes:
                    key = f"{symbol}_{timeframe}"

                    if key not in atrs:
                        atrs[key] = ATR(14)

                    for strategy_function in strategy_functions:
                        key = f"{symbol}_{timeframe}_{strategy_function}"

                        if key not in detectors:
                            try:
                                if strategy_function == "doubleBottomEntry":
                                    detectors[key] = DoubleBottomDetector(n_periods_to_confirm_swing=5, invalidation_n=72)
                                elif strategy_function == "doubleTopEntry":
                                    detectors[key] = DoubleTopDetector(n_periods_to_confirm_swing=5, invalidation_n=72)
                                elif strategy_function == "liqSweepEntry":
                                    detectors[key] = SweepDetector(n_periods_to_confirm_sweep=5, cross_pct_threshold=0.2)
                                else:
                                    logger.error(f"Unsupported strategy function: {strategy_function}")
                                    continue
                            except Exception as e:
                                logger.error(f"Error while initializing detector for key {key}: {e}")
                                continue

                            logger.debug(f"Initialized detector for {key}")

    except Exception as e:
        logger.error(f"Error while initializing detectors: {e}")

    return detectors, atrs

    


def object_cleanup(client: object, detectors=None, atrs=None):


    try:
        # Cleanup step
        current_symbols = set(symbol for strategy in client.config['strategies'] for symbol in strategy['symbols'])
    except Exception as e:
        logger.error("Error while getting current symbols: {0}".format(e))
        return

    try:
        keys_to_delete = []
        for key in detectors.keys():
            symbol, timeframe, _ = key.split("_")
            if symbol not in current_symbols:
                keys_to_delete.append(key)
    except Exception as e:
        logger.error("Error while generating keys to delete from detectors: {0}".format(e))
        return

    try:
        for key in keys_to_delete:
            del detectors[key]
            logger.debug(f"Removed detector for {key}")
    except Exception as e:
        logger.error("Error while deleting keys from detectors: {0}".format(e))
        return

    try:
        keys_to_delete = []
        for key in atrs.keys():
            symbol, timeframe = key.split("_")
            if symbol not in current_symbols:
                keys_to_delete.append(key)
    except Exception as e:
        logger.error("Error while generating keys to delete from atrs: {0}".format(e))
        return

    try:
        for key in keys_to_delete:
            del atrs[key]
            logger.debug(f"Removed atr for {key}")
    except Exception as e:
        logger.error("Error while deleting keys from atrs: {0}".format(e))
        return