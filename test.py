import json

with open("./data/test/data.json", "r") as f:
    # dict_keys(['qa', 'conversation', 'event_summary', 'observation', 'session_summary', 'sample_id']) 
    data = json.load(f)
    # data[0]["qa"][0]: {'question': 'When did Caroline go to the LGBTQ support group?', 'answer': '7 May 2023', 'evidence': ['D1:3'], 'category': 2}
"""
data[0]["conversation"].keys()
dict_keys(['speaker_a', 'speaker_b', 'session_1_date_time', 'session_1', 'session_2_date_time', 'session_2', 'session_3_date_time',
'session_3', 'session_4_date_time', 'session_4', 'session_5_date_time', 'session_5', 'session_6_date_time', 'session_6', 

data[0]["conversation"]["session_1"]

data[0]["conversation"]["session_1"][0]
{'speaker': 'Caroline', 'dia_id': 'D1:1', 'text': 'Hey Mel! Good to see you! How have you been?'}
"""

breakpoint()