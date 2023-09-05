import requests
import json
import os
import stix2
def fetch_mitre_data():
    api_url = 'https://cti-taxii.mitre.org/stix/collections/95ecc380-afe9-11e4-9b6c-751b66dd541e/objects/'
    headers = {
        'Accept': 'application/json',
        'User-Agent': 'my-application'
    }
    response = requests.get(api_url, headers=headers)
    print("Response Headers:", response.headers)
    print("Response Content:", response.content)
    
    if response.status_code == 200:
        mitre_data = response.json()
        file_path = os.path.join('data', 'report', 'mitre_data.json')
        with open(file_path, 'w') as f:
            json.dump(mitre_data, f, indent=4)
        print(f'MITRE data saved to {file_path}')
    else:
        print(f'Failed to fetch data: {response.status_code}')

fetch_mitre_data()



import json

def read_stix_file(file_path):
    with open('./data/report/mitre_data.json', 'w') as f:
        stix_data = json.load(f)
    return stix_data

