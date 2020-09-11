import requests

# 这是一个用于预训练的JSON请求
task_request = {
    "task_name": "test-datagen",  # Task的名称
    # 要用到的节点的列表
    "clients": [
        {
            "role": "main_client",  # 节点角色，目前有main_client, crypto_producer, feature_client, label_client 4种
            "addr": "127.0.0.1",  # 节点的IP地址
            "http_port": 8377,  # 节点的Http服务器端口（一般是固定的）
            # 节点要运行的Client的具体配置
            "client_config": {
                "client_type": "alignment_main",  # Client 类型，这里是样本对齐主节点
                "computation_port": 8378,  # Client进行MPC计算的RPC端口
            }
        },
        {
            "role": "feature_client",
            "addr": "127.0.0.1",
            "http_port": 8084,
            "client_config": {
                "client_type": "alignment_data",  # 样本对齐的数据节点，这种Client需要有额外的参数，直接加在下面即可。
                "computation_port": 8085,
                "raw_data_path": "Splitted_Indexed_Data/credit_default_data1.csv",  # 原始数据路径
                "out_data_path": "test-f1",  # 对齐后的数据路径（这里的路径并不是绝对路径，而是根据用户配置的某个根目录的相对路径），
                "columns": ["X1", "X2"]
            }
        },
        {
            "role": "feature_client",
            "addr": "127.0.0.1",
            "http_port": 8082,
            "client_config": {
                "computation_port": 8083,
                "client_type": "alignment_data",
                "raw_data_path": "Splitted_Indexed_Data/credit_default_data2.csv",
                "out_data_path": "test-f2"
            }
        },
        {
            "role": "label_client",
            "addr": "127.0.0.1",
            "http_port": 8884,
            "client_config": {
                "computation_port": 8885,
                "client_type": "alignment_data",
                "raw_data_path": "Splitted_Indexed_Data/credit_default_label.csv",
                "out_data_path": "test-l"
            }
        }
    ]
}
print(task_request)
# resp = requests.post("http://10.214.192.22:8380/createTask", json=task_request)
resp = requests.post("http://127.0.0.1:8380/createTask", json=task_request)

print(resp.status_code, resp.text)
