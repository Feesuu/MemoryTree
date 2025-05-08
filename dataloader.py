from config import globalconfig
import json
from collections import defaultdict

class Dataloder:
    def __init__(self, globalconfig):
        self.dataset_name = globalconfig.dataset_name
        self.data_path = f"./data/{globalconfig.dataset_name}/data.json"
        
        self.data = self.read_data()
    
    def read_data(self):
        data = []
        with open(self.data_path, "r") as f:
            raw_data = json.load(f)
        
        for item in raw_data:
            questions = item["qa"]
            conversation = item["conversation"]
            data_now = defaultdict(list)
            
            flag = True
            session_id = 1
            while flag:
                session_time_key = f"session_{session_id}_date_time"
                session_id_key = f"session_{session_id}"
                if session_id_key in conversation:
                    session_time = conversation[session_time_key]
                    conversation_list = conversation[session_id_key]
                    
                    conversation_data = list(
                        map(
                            lambda x: session_time + ":" + x["speaker"] + ":" + x["text"], 
                            conversation_list
                        )
                    )
                    
                    data_now[session_id_key] = conversation_data
                    session_id += 1
                else:
                    flag = False
            
            data.append((questions, data_now))
        return data
    
dataloader = Dataloder(globalconfig)
                
                