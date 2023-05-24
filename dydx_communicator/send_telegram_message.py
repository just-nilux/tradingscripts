import requests
import datetime
import pytz

def send_telegram_message(bot_token, chat_ids, text):
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
    

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Updater, CommandHandler, CallbackContext, CallbackQueryHandler
import threading
import requests
import datetime
import pytz
import logging

# other necessary imports...

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)
logger = logging.getLogger(__name__)



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
        [InlineKeyboardButton("Active Symbols", callback_data='1')],
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)

    update.message.reply_text('Please choose:', reply_markup=reply_markup)



def button(client):
    def inner_button(update: Update, context: CallbackContext) -> None:
        query = update.callback_query

        query.answer()

        if query.data == '1':
            if client.config.get('strategies') and 'symbols' in client.config['strategies'][0]:
                active_symbols = client.config['strategies'][0]['symbols']
                text = "\n".join(active_symbols)
                query.edit_message_text(text=f"Active Symbols: \n{text}")
            else:
                query.edit_message_text(text=f"No active symbols found.")

    return inner_button



def bot_main(bot_token: str, client) -> None:
    updater = Updater(token=bot_token, use_context=True)

    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CallbackQueryHandler(button(client)))

    updater.start_polling()

    updater.idle()
