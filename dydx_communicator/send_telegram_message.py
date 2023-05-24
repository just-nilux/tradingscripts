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
