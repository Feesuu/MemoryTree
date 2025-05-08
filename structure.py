import numpy as np
from typing import Dict, List, Optional, Set
from collections import defaultdict, deque
from utils import get_embedding, insert, calculate_cos, calculate_threshold, worker_ollama, update_vector, worker_openai, load_tree, save_tree
from config import globalconfig
from tqdm import tqdm
import multiprocessing

from multiprocessing import Pool, Array, Value
import ctypes
from typing import List, Dict

from prompt import AGGREGATE_PROMPT

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MemTreeNode:
    def __init__(self, content: str = ""):
        self.cv = content
        self.pv: Optional[int] = None            # 父节点ID（非对象，减少循环引用）
        self.dv: int = 0                         # 深度

class MemTree:
    def __init__(self, root_content: str = "Root"):
        self.root = MemTreeNode(root_content)
        self.nodes: Dict[int, MemTreeNode] = {id(self.root): self.root}  # 节点ID → 节点 
        self.adjacency: Dict[int, Set[int]] = defaultdict(set)           # 邻接表：父ID → 子ID-set
        self.size = 1

    # def add_node(self, content: str, parent_id: Optional[int] = None) -> int:
    #     """
    #     添加节点并返回新节点ID
    #     """
    #     parent_id = parent_id if parent_id else id(self.root)
    #     parent = self.nodes[parent_id]
        
    #     # node_ids_in_next_layer = list(map(lambda x: x, self.adjacency[parent_id]))
        
    #     # node_embs_in_next_layer = globalconfig.client .get(
    #     #     collection_name=globalconfig.collection_name,
    #     #     ids=node_ids_in_next_layer,
    #     #     output_fields=["vector"]
    #     # )
    #     # breakpoint()

    #     new_node = MemTreeNode(content)
    #     new_node.pv = parent_id
    #     new_node.dv = parent.dv + 1
    #     new_id = id(new_node)
        
    #     # breakpoint()
    #     # embdding & insert into vdb
    #     if content:
    #         ev = get_embedding(content)
    #         ev = ev.flatten().tolist()
    #         # breakpoint()
    #         insert([{"id": new_id, "vector": ev}])

    #     self.nodes[new_id] = new_node
    #     self.adjacency[parent_id].add(new_id) # 更新父节点的邻接表
    #     self.size += 1
    #     return new_id
    
    def add_node_single(self, content: str, ev: np.ndarray, current_parent_id: int):
        current_parent = self.nodes[current_parent_id]
        
        new_node = MemTreeNode(content)
        new_node.pv = current_parent_id
        new_node.dv = current_parent.dv + 1
        new_id = id(new_node)
        
        # insert into vdb
        ev = ev.flatten().tolist()
        insert([{"id": new_id, "vector": ev}])

        self.nodes[new_id] = new_node
        self.adjacency[current_parent_id].add(new_id)
        self.size += 1
        
        return new_id
    
    def add_node(self, content: str, parent_id: Optional[int] = None) -> int:
        current_parent_id = parent_id if parent_id else id(self.root)
        ev = get_embedding(content) # 1*D
        parent_ids = []
        
        is_continue_traversal = True
        while is_continue_traversal:
            
            current_parent = self.nodes[current_parent_id]
            current_depth = current_parent.dv + 1
            # list of node_id
            node_ids_in_next_layer = list(map(lambda x: x, self.adjacency[current_parent_id]))
            #breakpoint()
            
            if not node_ids_in_next_layer:
                break
            # list of dict_keys(['id', 'vector'])
            node_results_in_next_layer = globalconfig.client.get(
                collection_name=globalconfig.collection_name,
                ids=node_ids_in_next_layer,
            )
            
            # shape: node_nums * dimension
            node_embs_in_next_layer = np.array(
                list(map(lambda x: x["vector"], node_results_in_next_layer))
            )
            
            try:
                cos = calculate_cos(v=ev, M=node_embs_in_next_layer)
            except:
                breakpoint()
            current_threshold = calculate_threshold(current_depth=current_depth)
            cos = (cos > current_threshold).astype(int) * cos
            max_cos = np.max(cos)
            # if content == "computer":
            #     breakpoint()
        
            if max_cos:
                # Continue to traverse
                max_index = int(np.argmax(cos))
                current_parent_id = node_ids_in_next_layer[max_index]
                parent_ids.append(current_parent_id)
            else:
                # all child nodes' similarities are below the threshold
                # v_new is directly attached as a new leaf node under the current node.
                is_continue_traversal = False
        
        new_id = self.add_node_single(content, ev, current_parent_id)
        
        # if content == "computer":
        #     breakpoint()
        self.modify_nodes(new_content=content, node_ids=parent_ids)
        
        return new_id
        # return new_id, current_parent_id

    # def get_children(self, node_id: int) -> List[MemTreeNode]:
    #     """
    #     根据邻接表获取子节点对象列表
    #     """
    #     return [self.nodes[child_id] for child_id in self.adjacency.get(node_id)]

    # def traverse_from_root(self) -> List[MemTreeNode]:
    #     """
    #     迭代遍历（BFS）
    #     """
    #     from collections import deque
    #     visited = []
    #     queue = deque([id(self.root)])
    #     while queue:
    #         node_id = queue.popleft()
    #         visited.append(self.nodes[node_id])
    #         queue.extend(list(self.adjacency.get(node_id)))
    #     return visited
    
    def print_tree_terminal(self, max_depth: int = 3):
        """在终端按层级打印树结构"""
        if not self.nodes:
            print("(空树)")
            return
        
        # 使用BFS队列: (节点ID, 节点对象)
        queue = deque([(id(self.root), self.root)])
        
        while queue:
            node_id, node = queue.popleft()
            
            # 打印当前节点
            indent = "    " * node.dv
            parent_info = f" → 父[{node.pv}]" if node.pv else ""
            print(f"{indent}├─ ID:{node_id} 内容:'{node.cv}' 深度:{node.dv}{parent_info}")
            
            # # 如果达到最大深度则停止
            # if node.dv >= max_depth:
            #     continue
                
            # 添加子节点到队列
            for child_id in self.adjacency[node_id]:
                if child_id in self.nodes:
                    child = self.nodes[child_id]
                    #child.dv = node.dv + 1  # 更新子节点深度
                    queue.append((child_id, child))

    def modify_nodes(self, new_content: str, node_ids: List[int]):
        if node_ids:
            # 预先分配共享内存
            total_tasks = len(node_ids)
            # max_length = max(len(node.cv) for node in self.nodes.values()) + 20
            # shared_cvs = [Array(ctypes.c_char, max_length) for _ in range(total_tasks)]
            # shared_len_children = Array(ctypes.c_int, total_tasks)
            
            # # 初始化共享数据
            # for i, node_id in enumerate(node_ids):
            #     if node_id in self.nodes:
            #         node = self.nodes[node_id]
            #         shared_cvs[i].value = node.cv.encode()
            #         shared_len_children[i] = len(self.adjacency[node_id])
            
            # tasks = [(node_id, shared_cvs[i], shared_len_children[i], new_content) for i, node_id in enumerate(node_ids)]
            
            tasks = [(node_id, self.nodes[node_id].cv, len(self.adjacency[node_id]), new_content) for i, node_id in enumerate(node_ids)]
            
            update_nodes = deque()
            with tqdm(total=total_tasks, desc="Processing updation of parent nodes traversed along the path...") as pbar:
                with multiprocessing.Pool(processes=globalconfig.llm_parallel_nums) as pool:
                    for result in pool.imap_unordered(self._modify_shared_mem, tasks):
                        node_id, output = result
                        if output is not None:
                            update_nodes.append((node_id, output))
                            pbar.update(1)
            
            #更新之前，你讲之前的node信息，加入到当前节点中
            #banana+apple->fruit, apple挂到ftuit下。
            
            content_from_origin_node = []
            content_from_current_node = []
            for item in update_nodes:
                node_id, update_content = item
                # save content from original nodes and current nodes
                content_from_origin_node.append(self.nodes[node_id].cv)
                content_from_current_node.append(update_content)
                self.nodes[node_id].cv = update_content
                
            # Batch
            evs_from_origin_node = get_embedding(content_from_origin_node, batch=globalconfig.embedding_batch_size)
            evs_from_current_node = get_embedding(content_from_current_node, batch=globalconfig.embedding_batch_size)
            evs_from_current_node = evs_from_current_node.tolist()
            
            # Update current node
            update_data = [
                {"id": update_nodes[i][0], "vector": evs_from_current_node[i]} for i in range(total_tasks)
            ]
            update_vector(new_data=update_data)
            
            # Insert original node
            # for i in range(total_tasks):
            self.add_node_single(content=content_from_origin_node[-1], ev=evs_from_origin_node[-1], current_parent_id=update_nodes[-1][0])
    
    @staticmethod
    def _modify_shared_mem(args):
        node_id, current_content, len_children, new_content = args
        # current_content = cv.value.decode()
        
        input_prompt = AGGREGATE_PROMPT.format(
            new_content=new_content,
            n_children=str(len_children), 
            current_content=current_content,
        )
        
        output = worker_ollama(input_prompt)
        
        return node_id, output
        # cv.value = (current + " modified").encode()


def build_tree(data):
    # single data source
    tree = load_tree(globalconfig.save_path)
    questions, sessions = data.data[0]
    
    if tree is None:
        tree = MemTree("")
        root_id = id(tree.root)
        
        all_sessions = []
        for session_id, session in sessions.items():
            all_sessions.extend(session)
            break # 目前单session测试
        
        for session in all_sessions:
            tree.add_node(session, root_id)

        save_tree(tree, globalconfig.save_path)
        
    return tree