import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get('ALIENVAULT_API_KEY')

# API endpoint for AlienVault OTX
API_ENDPOINT = 'https://otx.alienvault.com/api/v1/indicators/domain/google.com'

# Headers for the API request
headers = {
    'X-OTX-API-KEY': API_KEY
}

# Make the API request
response = requests.get(API_ENDPOINT, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    
    # Create directory if it doesn't exist
    if not os.path.exists('./data/alienVault'):
        os.makedirs('./data/alienVault')
    
    # Save the data to a JSON file
    with open('./data/alienVault/data.json', 'w') as f:
        json.dump(data, f, indent=4)
    print('Data saved successfully.')
else:
    print(f'Failed to fetch data. Status code: {response.status_code}')