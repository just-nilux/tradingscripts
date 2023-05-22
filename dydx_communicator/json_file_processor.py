import json
import hashlib
import os
from collections import defaultdict

def calculate_file_hash(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return None

    # Read the file content and calculate the hash
    with open(file_path, 'rb') as f:
        content = f.read()
        file_hash = hashlib.md5(content).hexdigest()

    return file_hash


def process_json_file(file_path):
    # Calculate the current hash value of the file
    current_hash = calculate_file_hash(file_path)

    # Check if the file has been updated since the last execution
    if current_hash == process_json_file.last_hash:
        print("File has not been updated since the last execution.")
        return

    # Update the last hash with the current hash
    process_json_file.last_hash = current_hash

    # Open and load the json data
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Create a dictionary to hold the results, with default value as an empty list
    results = defaultdict(list)

    # Loop through the data and collect prices for each symbol
    for res in data['dydx']:
        if res['type'] == 'LineToolHorzLine':
            symbol = res['symbol']
            price = res['points'][0]['price']  # assuming there is always at least one point
            results[symbol].append(price)

    # Print the results
    for symbol, prices in results.items():
        rounded_prices = [round(price, 2) for price in prices]  # Round the prices
        print(f"Symbol: {symbol}, Prices: {rounded_prices}")
    
    return results


# Initialize the last hash as an empty string
process_json_file.last_hash = ''

# Example usage
json_file_path = '/opt/tvserver/database.json'
result = process_json_file(json_file_path)
