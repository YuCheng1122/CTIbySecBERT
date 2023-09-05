import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.environ.get('VIRUSTOTAL_API_KEY')
URL = "https://www.virustotal.com/api/v3/domains/{domain}"

headers = {
    "x-apikey": API_KEY
}

def get_domain_info(domain):
    response = requests.get(URL.format(domain=domain), headers=headers)
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None

if __name__ == '__main__':
    domains = ['alexa.com', 'phishtank.com', 'github.com','malwaredomainlist.com','virustotal.com','openphish.com']  
    for domain in domains:
        info = get_domain_info(domain)
        if info:
            data_path = os.path.join('../../data/domain', f'{domain}.json')
            with open(data_path, 'w') as f:
                json.dump(info, f, indent=4)
            print(f'Data saved to {data_path}')
