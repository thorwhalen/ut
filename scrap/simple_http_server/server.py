# server.py

import json

data = [{'title': 'Hello World!', 'body': "Your big, blue, roundness impresses us all."}]
json.dump(data, open('demo_data.json', 'w'))


