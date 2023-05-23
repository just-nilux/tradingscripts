import requests

def send_telegram_message(bot_token, chat_ids, text):
    for chat_id in chat_ids:
        send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={text}'
        response = requests.get(send_text)
        return response.json()
