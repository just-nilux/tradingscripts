from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackContext, CallbackQueryHandler

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram import Update
from logger_setup import setup_logger

import threading
import requests
import datetime
import pytz
import logging

logger = setup_logger(__name__)


def send_telegram_message(bot_token: str, chat_ids: list, text: str):
    # get current time in Copenhagen timezone
    now = datetime.datetime.now(pytz.timezone('Europe/Copenhagen'))
    current_hour = now.hour
    current_minute = now.minute

    # Define the excluded nighttime and evening ranges
    is_nighttime = 23 <= current_hour or current_hour < 6
    is_evening = 18 <= current_hour < 20 or (current_hour == 20 and current_minute <= 30)

    # if the current hour is within the allowed time window and the minute is 0, send the messages
    if not is_nighttime and not is_evening and current_minute == 0:
        for chat_id in chat_ids:
            send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={text}'
            response = requests.get(send_text)
        return response.json()

    # if the current hour is outside the allowed time window or it's not the start of the hour, don't send any messages
    else:
        print("Message not sent due to it being outside the allowed time windows (23:00 - 06:00 and 18:00 - 20:30) or not at the top of the hour.")
        return None




def start(update: Update, context: CallbackContext) -> None:
    
    keyboard = [
        ["Active Symbols"],
        ["Active Strategies"]
    ]

    reply_markup = ReplyKeyboardMarkup(keyboard)

    update.message.reply_text('Please choose:', reply_markup=reply_markup)



def process_response(update: Update, context: CallbackContext):
    client = context.bot_data['client']

    response = update.message.text

    if response == 'Active Symbols':
        if client.config.get('strategies'):
            strategy = client.config['strategies'][0]
            if 'symbols' in strategy and 'timeframes' in strategy:
                active_symbols = strategy['symbols']
                timeframes = strategy['timeframes']
                messages = []
                for symbol in active_symbols:
                    message = f"Symbol: {symbol} - Timeframes: {', '.join(timeframes)}"
                    messages.append(message)
                text = "\n".join(messages)
                update.message.reply_text(text=f"Active Symbols: \n{text}")
            else:
                update.message.reply_text(text=f"No active symbols or timeframes found.")
                
    elif response == 'Active Strategies':
        if client.config.get('strategies'):
            strategy = client.config['strategies'][0]
            if 'strategy_functions' in strategy:
                active_strategies = strategy['strategy_functions']
                messages = []
                for strategy_func in active_strategies:
                    message = f"- {strategy_func.replace('_', ' ').title()}"
                    messages.append(message)
                text = "\n".join(messages)
                update.message.reply_text(text=f"Active Strategies: \n{text}")
            else:
                update.message.reply_text(text=f"No active strategies found.")



def bot_main(bot_token: str, client) -> None:
    updater = Updater(token=bot_token, use_context=True)

    # Store the client in context
    updater.dispatcher.bot_data['client'] = client

    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), process_response))

    updater.start_polling()

    updater.idle()

