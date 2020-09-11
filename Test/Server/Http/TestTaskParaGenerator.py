import json
from Server.HttpServer.TaskParaGenerator import generate_task_paras, generate_dataset_json

task_request = {
    "task_name": "test-datagen",
    "clients": [
        {
            "role": "main_client",
            "addr": "127.0.0.1",
            "http_port": 8377,
            "client_config": {
                "client_type": "alignment_main",
                "computation_port": 8378,
                "in_dim": 64,
                "out_dim": 1,
                "layers": [1],
                "batch_size": 64,
                "test_batch_size": 10000,
                "test_per_batches": 1001,
                "learning_rate": 0.1,
                "max_iter": 10010,
            }
        },
        {
            "role": "feature_client",
            "addr": "127.0.0.1",
            "http_port": 8084,
            "client_config": {
                "client_type": "alignment_client",
                "computation_port": 8085,
                "raw_data_path": "Splitted_Indexed_Data/credit_default_data1.csv",
                "out_data_path": "test-f1"
            }
        },
        {
            "role": "feature_client",
            "addr": "127.0.0.1",
            "http_port": 8082,
            "client_config": {
                "computation_port": 8083,
                "client_type": "alignment_client",
                "raw_data_path": "Splitted_Indexed_Data/credit_default_data2.csv",
                "out_data_path": "test-f1"
            }
        },
        {
            "role": "feature_client",
            "addr": "127.0.0.1",
            "http_port": 8884,
            "client_config": {
                "computation_port": 8885,
                "client_type": "alignment_client",
                "raw_data_path": "Splitted_Indexed_Data/credit_default_label.csv",
                "out_data_path": "test-f1"
            }
        }
    ]
}

# task_request = {
#     "task_name": "test-secure_xgb",
#     "clients": [
#         {
#             "role": "main_client",
#             "addr": "127.0.0.1",
#             "http_port": 8377,
#             "client_config": {
#                 "client_type": "secure_xgboost_main",
#                 "computation_port": 8378,
#                 "metric": "auc_ks",
#                     "config": {
#                         "learning_rate": 0.1,
#                         "sync_info": {
#                             "seed": 8964
#                         },
#                         "max_iteration": 5,
#                         "max_depth": 4,
#                         "reg_lambda": 1,
#                         "gamma": 0.,
#                         "col_sample_ratio": 0.8,
#                         "row_sample_ratio": 1.,
#                         "batch_size": None,
#                         "test_batch_size": None
#                     }
#             }
#         },
#         {
#             "role": "feature_client",
#             "addr": "127.0.0.1",
#             "http_port": 8084,
#             "client_config": {
#                 "client_type": "secure_xgboost_feature",
#                 "computation_port": 8085,
#
#                 "data_path": "Splitted_Indexed_Data/credit_default_data1.csv"
#             }
#         },
#         {
#             "role": "feature_client",
#             "addr": "127.0.0.1",
#             "http_port": 8082,
#             "client_config": {
#                 "computation_port": 8083,
#                 "client_type": "secure_xgboost_feature",
#                 "data_path": "Splitted_Indexed_Data/credit_default_data2.csv"
#             }
#         },
#         {
#             "role": "label_client",
#             "addr": "127.0.0.1",
#             "http_port": 8884,
#             "client_config": {
#                 "computation_port": 8885,
#                 "client_type": "secure_xgboost_label",
#                 "data_path": "Splitted_Indexed_Data/credit_default_label.csv",
#             }
#         }
#     ]
# }



if __name__ == '__main__':
    client_paras, client_addrs = generate_task_paras(task_request)
    print(client_addrs)
    # for client_para in client_paras:
        # print(json.dumps(client_para, indent=2))
    client_data = generate_dataset_json(task_request)
    print(json.dumps(client_data, indent=2))