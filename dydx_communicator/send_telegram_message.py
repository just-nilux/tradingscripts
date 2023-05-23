import requests

def send_telegram_message(bot_token, chat_ids, text):
    for chat_id in chat_ids:
        send_text = f'https://api.telegram.org/bot{bot_token}/sendMessage?chat_id={chat_id}&parse_mode=Markdown&text={text}'
        response = requests.get(send_text)
        return response.json()

# Usage:
bot_token = '5951325673:AAGvqkhnogQGZPsg8B72xChi5FRsjTEbeDk'
chat_ids = ['470493359', 'chat_id_2', 'chat_id_3']  # Add chat ids as string here
text = 'Hello, World!'

send_telegram_message(bot_token, chat_ids, text)
