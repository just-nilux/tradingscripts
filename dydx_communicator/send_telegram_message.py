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
import json

logger = setup_logger(__name__)



with open('config.json') as config_file:
    config = json.load(config_file)



def send_large_message(update, text, limit=4096):
    while len(text) > 0:
        if len(text) > limit:
            cut_index = text[:limit].rfind('\n')
            if cut_index == -1:
                cut_index = limit
            update.message.reply_text(text=text[:cut_index])
            text = text[cut_index:]
        else:
            update.message.reply_text(text=text)
            text = ''



def send_telegram_message(text: str, pass_time_limit = False):
    bot_token = config['bot_token']
    chat_ids = config['chat_ids']
    now = datetime.datetime.now(pytz.timezone('Europe/Copenhagen'))
    current_hour = now.hour
    current_minute = now.minute
    is_nighttime = 23 <= current_hour or current_hour < 6
    is_evening = 18 <= current_hour < 20 or (current_hour == 20 and current_minute <= 30)

    if not is_nighttime and not is_evening and current_minute == 0 or pass_time_limit:
        for chat_id in chat_ids:
            send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={text}'
            response = requests.get(send_text)
        return response.json()

    else:
        logger.info("Message not sent due to it being outside the allowed time windows (23:00 - 06:00 and 18:00 - 20:30) or not at the top of the hour.")
        return None



def start(update: Update, context: CallbackContext) -> None:
    keyboard = [
        ["Active Symbols", "Active Strategies"],
        ["Open Positions", "Open Orders"],
        ['Account Equity', "Algo Stats"]
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
                if not active_symbols:
                    update.message.reply_text(text=f"No Active Symbols ATM.")
                    return
                timeframes = strategy['timeframes']
                messages = []
                for symbol in active_symbols:
                    message = f"Symbol: {symbol} \nTimeframes: {', '.join(timeframes)}\n"
                    messages.append(message)
                text = "\n".join(messages)
                send_large_message(update, text=f"*** Active Symbols: ***\n{text}")

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
                send_large_message(update, text=f"*** Active Strategies: ***\n{text}")
            else:
                update.message.reply_text(text=f"No active strategies found.")

    elif response == 'Open Positions':
        positions = client.fetch_all_open_position()
        send_large_message(update, text=positions)

    elif response == 'Account Equity':
        acc_equity = client.fetch_free_equity()
        if acc_equity is not None:
            update.message.reply_text(text=f"Account Equity: {round(acc_equity,1)}$ ")
        else:
            update.message.reply_text(text="Failed to fetch account equity.")

    elif response == 'Open Orders':
        orders = client.get_open_orders()['orders']
        if orders:
            messages = []
            for order in orders:
                trigger_price = f"\nTrigger Price: {order['triggerPrice']}" if order['triggerPrice'] is not None else ""
                message = f"Symbol: {order['market']}\nType: {order['market']}\nOrder Type: {order['type']}\nPrice: {order['price']}{trigger_price}\nSize: {order['size']}\n"
                messages.append(message)
            text = "\n".join(messages)
            send_large_message(update, text=f"*** Open Orders: ***\n\n{text}")
        else:
            update.message.reply_text(text="No open orders.")

    elif response == "Algo Stats":
        update.message.reply_text(text="Coming soon......")



def bot_main(bot_token: str, client) -> None:
    updater = Updater(token=bot_token, use_context=True)
    updater.dispatcher.bot_data['client'] = client
    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), process_response))
    updater.start_polling()
    updater.idle()
