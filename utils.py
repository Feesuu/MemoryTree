from tqdm import tqdm
from config import globalconfig
import re
import math
import os
from collections import deque
from openai import OpenAI
from hashlib import md5
import multiprocessing
import numpy as np
from typing import Dict, List, Optional, Set, Union, Tuple

def get_embedding(texts, batch=1):
    texts_embeddings = globalconfig.model.encode(texts, convert_to_tensor=True, show_progress_bar=True, device="cuda", batch_size=batch)
    texts_embeddings = texts_embeddings.cpu().numpy()
    
    if texts_embeddings.ndim == 1:
        texts_embeddings = texts_embeddings.reshape(1, -1)
    return texts_embeddings
    
def insert(data):
    globalconfig.client.insert(collection_name=globalconfig.collection_name, data=data)
    
def batch_insert(data, BATCH_SIZE):
    total = len(data)
    with tqdm(total=total, desc=f"Inserting into {globalconfig.collection_name}") as pbar:
        for i in range(0, total, BATCH_SIZE):
            batch = data[i:i+BATCH_SIZE]
            globalconfig.client.insert(collection_name=globalconfig.collection_name, data=batch)
            pbar.update(len(batch))     
            
def search(query:list[list[float]], output_fields=None, top_k=None, filter=None):
    if filter:
        res = globalconfig.client.search(
            collection_name=globalconfig.collection_name,
            data=query,
            limit=top_k,
            output_fields=output_fields,
            filter=filter,
        )
    else:
        res = globalconfig.client.search(
            collection_name=globalconfig.collection_name,
            data=query,
            limit=top_k,
            output_fields=output_fields,
        )
    return res

def calculate_threshold(current_depth):
    threshold = globalconfig.base_threshold * math.exp(globalconfig.rate * current_depth / globalconfig.max_depth)
    return threshold

from sklearn.metrics.pairwise import cosine_similarity
def calculate_cos(v, M):
    return cosine_similarity(v, M).flatten()


import ollama
from pathlib import Path
from tqdm import tqdm

def worker_ollama(prompt):
    """ 包装函数用于处理异常 """
    # question_id, prompt = args[0], args[1]
    try:
        client = ollama.Client(host="http://localhost:5001/forward")
        res = client.chat(
            model="llama3.1:8b4k",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        return res.message.content
    except Exception as e:
        print(f"Error processing {str(e)}")
        return None
    
def update_vector(new_data):
    globalconfig.client.upsert(
        collection_name=globalconfig.collection_name,
        data=new_data,
    )
    
def mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()
    
def retrieve_single(args: Tuple[str, List[np.ndarray]]):
    query_id, query_emb = args
    relevent_contexts = search(query_emb, top_k=globalconfig.top_k_retrieve)
    relevent_contexts = relevent_contexts[0]
    #breakpoint()
    res = list(map(lambda x: x["id"], relevent_contexts))
    
    return query_id, res
    
def retrieve(query: list[str]):
    query_embeddings = get_embedding(query, globalconfig.embedding_batch_size)
    #query_ids = list(map(lambda x: mdhash_id(x), query))
    
    total_tasks = len(query)
    tasks = [(query[i], [query_embeddings[i]]) for i in range(total_tasks)]
    update = []
    # with tqdm(total=total_tasks, desc="Processing query retrieve...") as pbar:
    #     with multiprocessing.Pool(processes=globalconfig.llm_parallel_nums) as pool:
    #         for result in pool.imap_unordered(retrieve_single, tasks):
    #             query_id, res = result
    #             update.append((query_id, res))
    #             pbar.update(1)
    
    for task in tasks:
        update.append(retrieve_single(task))
    return update
    
import pickle
def save_tree(tree, filename='memtree.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(tree, f)

def load_tree(filename='memtree.pkl'):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

from prompt import ANSWER_PROMPT
def generation(tree, retrieve_results):
    
    results = []
    for que, contexts_id in retrieve_results:
        contexts = list(map(lambda x: tree.nodes[x].cv, contexts_id))
        # breakpoint()
        contexts = "\n\n".join(contexts)
        prompt = ANSWER_PROMPT.format(query=que, retrieved_content=contexts)
        
        output = worker_ollama(prompt)
        results.append((que, contexts, output))
        
    return results


        
        



        
    