from config import globalconfig
from utils import retrieve, generation
from dataloader import dataloader
from structure import build_tree

if __name__ == "__main__":
    
    tree = build_tree(dataloader)
    questions = dataloader.data[0][0]
    # 筛选D1的
    d1_query = []
    for item in questions:  
        if item["evidence"] and "answer" in item:
            if "D1:" in item["evidence"][0]:
                try:
                    d1_query.append((item["question"], item["answer"]))
                except:
                    breakpoint()
    
    questions = list(map(lambda x: x[0], d1_query))
    
    retrieve_result = retrieve(questions)
    
    print(retrieve_result)
    
    result = generation(tree, retrieve_result)
    
    breakpoint()
    
    
    