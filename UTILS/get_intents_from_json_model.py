# Get the list of intents from the exported LUIS.ai model (json)

import json
import csv
from pathlib import Path

json_path = r"D:\OneDrive - UTS\36102 iLab 1 - Spring 2019\MODEL\9f86f658-f7a2-44c1-a25a-1233ae7f9e84_v0.1.json"

# %%
with open(json_path, 'r') as infile:
    model = json.load(infile)
    
intent_list = [d['name'] for d in model['intents']]

out_path = 
with open()