from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import re
import logging


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def error_handler(update: Update, context: CallbackContext):
    """Log the error and send a telegram message to notify the developer."""
    # Log the error before we do anything else, so we can see it even if something breaks.
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    # Now send the message (optionally to yourself, for logging purposes)
    if update.effective_message:
        text = f"Hey.\n The error {context.error} occurred in the context of this message:\n{update.effective_message}"
        update.effective_message.reply_text(text)



def check_close_trade(message, original_message):
    trade_info = {}

    if any(word in message.lower() for word in ['lukker', 'lukke', 'lukkes']) and any(word in message.lower() for word in ['traden', 'trade']):
        coin_match = re.search(r'COIN:\s*([A-Z]+)', original_message, re.IGNORECASE)
        trade_info['close_trade'] = coin_match.group(1) + "USD"
        return trade_info
    return None



def extract_trade_info(message):
    trade_info = {}

    coin_match = re.search(r'COIN:\s*([A-Z]+)', message, re.IGNORECASE)
    if coin_match:
        trade_info['symbol'] = coin_match.group(1) + "USD"

    trade_type_match = re.search(r'TRADE TYPE\s*-\s*(\w+)', message, re.IGNORECASE)
    if trade_type_match:
        trade_info['side'] = trade_type_match.group(1)

    entry_match = re.search(r'Min entry:\s*([\d.]+)', message, re.IGNORECASE)
    if entry_match:
        trade_info['entry'] = float(entry_match.group(1))

    target1_match = re.search(r'Mit target 1:\s*([\d.]+)', message, re.IGNORECASE)
    if target1_match:
        trade_info['take_profit'] = float(target1_match.group(1))

    sl_match = re.search(r'Mit SL:\s*([\d.]+)', message, re.IGNORECASE)
    if sl_match:
        trade_info['stop_loss'] = float(sl_match.group(1))

    timeframe_match = re.search(r'TIMEFRAME:\s*(\w+)', message, re.IGNORECASE)
    if timeframe_match:
        trade_info['timeframe'] = timeframe_match.group(1)

    # Check if all keys are present
    required_keys = ['symbol', 'side', 'entry', 'take_profit', 'stop_loss', 'timeframe']
    if all(key in trade_info for key in required_keys):
        return trade_info
    else:
        return None



def handle_message(update: Update, context: CallbackContext) -> None:
    message = update.message.text



    # If this message is a reply to another message
    if update.message.reply_to_message is not None:
        original_message = update.message.reply_to_message.caption
        result = check_close_trade(message, original_message)
        if result is not None:
            context.bot.send_message(chat_id=-987354457, text=str(result))
            print(f"{result}")

    trade_info = extract_trade_info(message)
    if trade_info is not None:
        context.bot.send_message(chat_id=-987354457, text=str(trade_info))
        print(trade_info)



def main() -> None:
    # Use your own bot token here
    updater = Updater("5828078593:AAEDwdObGT1JFqhCXRyH1wUZO0_FpGrHnNk")

    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_message))
    dp.add_error_handler(error_handler)  # Register the error handler


    # Start the Bot
    updater.start_polling()

    # Run the bot until the user presses Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT
    updater.idle()

if __name__ == '__main__':
    main()



