
import argparse
from pymilvus import MilvusClient
import html
import re

def clean_str(input) -> str:
    """Clean an input string by removing HTML escapes, control characters, and other unwanted characters."""
    # If we get non-string input, just give it back
    if not isinstance(input, str):
        return input

    result = html.unescape(input.strip())
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)

    # Remove non-alphanumeric characters and convert to lowercase
    return re.sub('[^A-Za-z0-9 ]', ' ', result.lower()).strip()

def get_embedding_model(config):
    from sentence_transformers import SentenceTransformer
    
    if config.embedding_model_name == "models--BAAI--bge-m3":
        model = SentenceTransformer("/home/yaodong/codes/GNNRAG/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181")
        
    elif config.embedding_model_name == "nvidia/NV-Embed-v2":
        model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True, model_kwargs={"torch_dtype": "float16"}, device="cuda")
        model.max_seq_length = 4096
        model.tokenizer.padding_side="right"
    else:
        pass
    
    return model

def create_collections(client, collection_name, dimension=1024):
    # if client.has_collection(collection_name=collection_name):
    #     client.drop_collection(collection_name=collection_name)
    if client.has_collection(collection_name=collection_name):
        print(f"Collection '{collection_name}' already exists, using existing collection")
        return 

    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
    )
    return
    
# def create_collections(client, collection_name, dimension=1024):
#     from pymilvus import MilvusClient, DataType
    
#     # 检查集合是否已存在
#     if client.has_collection(collection_name=collection_name):
#         print(f"Collection '{collection_name}' already exists, using existing collection")
#         return 
    
#     # 创建新集合
#     schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=True)
    
#     schema.add_field(
#         field_name="id",
#         datatype=DataType.VARCHAR,
#         is_primary=True,
#         auto_id=True,
#         max_length=100
#     )
    
#     schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dimension)
    
#     index_params = MilvusClient.prepare_index_params()
#     index_params.add_index(
#         field_name="vector",
#         index_type="AUTOINDEX", 
#         metric_type="COSINE"
#     )
    
#     # 创建集合和索引
#     client.create_collection(
#         collection_name=collection_name,
#         dimension=dimension,
#         schema=schema,
#     )
#     client.create_index(collection_name=collection_name, index_params=index_params)
    
#     print(f"Created new collection '{collection_name}' with dimension {dimension}")
#     return 

def load_config(config_path, args):
    import yaml
    from types import SimpleNamespace
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    f.close()
    
    # updata key
    for key, value in vars(args).items():
        if key not in config: 
            config[key] = value
        
    config = SimpleNamespace(**config)
    
    return config

class GlobalConfig:
    def __init__(self, config):
        # 将 config 的所有键值动态绑定到当前实例
        for key, value in vars(config).items():
            setattr(self, key, value)  # 相当于 self.key = value
           
        self.model= get_embedding_model(config)
        self.db_name = f'./data/{config.dataset_name}/{clean_str(config.embedding_model_name).replace(" ", "")}_{config.vdb_name}'
        self.client = MilvusClient(self.db_name)
        create_collections(self.client, config.collection_name, config.dimension)
        self.save_path = f"./data/{config.dataset_name}/{config.save_name}"

parser = argparse.ArgumentParser(description='Path to config')
parser.add_argument('--config_path', type=str,default="./config/config.yaml")
args = parser.parse_args()
config_yaml = load_config(config_path=args.config_path, args=args)

globalconfig = GlobalConfig(config_yaml)
        
        
        