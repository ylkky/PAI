import numpy as np
import threading
import traceback
from Client.Client import BaseClient
from Communication.Message import MessageType, ComputationMessage
from Communication.Channel import BaseChannel
from Client.Data import DataLoader
from Utils.Log import Logger


class TripletsProvider(BaseClient):
    """
    A client for generate beaver triples. It can listen to other clients
    """
    def __init__(self, client_id: int, channel: BaseChannel):
        super(TripletsProvider, self).__init__(client_id, channel)
        self.triplets_id = self.client_id
        self.triplet_proposals = dict()
        self.listening_thread = [None for _ in range(channel.n_clients)]
        self.listening = False

    def listen_to(self, sender: int):
        msg = self.receive_msg(sender)
        if msg is not None:
            if msg.header == MessageType.SET_TRIPLET:
                target = msg.data[0]
                shape_sender = msg.data[1]
                shape_target = msg.data[2]
                existing_shapes = self.triplet_proposals.get((target, sender))
                if existing_shapes is not None:
                    if existing_shapes[0] == shape_target and existing_shapes[1] == shape_sender:
                        self.generate_and_send_triplets((sender, target), (shape_sender, shape_target))
                        del self.triplet_proposals[(target, sender)]
                    else:
                        self.logger.logW("Triplet shapes %s %s not match with clients %d and %d" % (shape_target, shape_sender, target, sender))
                else:
                    self.triplet_proposals[(sender, target)] = (shape_sender, shape_target)
            else:
                self.logger.logW("Expect SET_TRIPLET message, but received %s from %d" % (msg.header, sender))

    def generate_and_send_triplets(self, clients, shapes):
        # 判断哪一个是乘数，哪一个是被乘数
        if shapes[1][0] != shapes[0][1]:
            shapes = [shapes[1], shapes[0]]
            clients = [clients[1], clients[0]]
        u0 = np.random.uniform(-1, 1, shapes[0])
        u1 = np.random.uniform(-1, 1, shapes[0])
        v0 = np.random.uniform(-1, 1, shapes[1])
        v1 = np.random.uniform(-1, 1, shapes[1])
        z = np.matmul(u0 + u1, v0 + v1)
        z0 = z * np.random.uniform(0, 1, z.shape)
        z1 = z - z0
        self.send_msg(clients[0], ComputationMessage(MessageType.TRIPLE_ARRAY, (clients[1], u0, v0, z0)))
        self.send_msg(clients[1], ComputationMessage(MessageType.TRIPLE_ARRAY, (clients[0], v1, u1, z1)))

    def listen_to_client(self, sender_id):
        while self.listening:
            self.listen_to(sender_id)

    def start_listening(self):
        """
        Start the listening thread
        """
        self.listening = True
        for i in range(self.channel.n_clients):
            self.listening_thread[i] = threading.Thread(target=self.listen_to_client, args=(i,))
            self.listening_thread[i].start()

    def stop_listening(self):
        """
        Stop the listening thread
        """
        self.listening = False
        for i in range(self.channel.n_clients):
            self.listening_thread[i].join()


class DataClient(BaseClient):
    def __init__(self, client_id: int, channel: BaseChannel, dataloader: DataLoader, batch_size: int, data_dim: int, output_dim: int,
                 triplets_id: int=None):
        # random generate some to data
        super(DataClient, self).__init__(client_id, channel)
        self.dataloader =dataloader
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.data_dim = data_dim
        self.para = np.random.uniform(-1, 1, [data_dim, output_dim])
        self.batch_data = None
        self.triplets_id = triplets_id


        # 变量储存器，用于Secret Sharing矩阵乘法
        self.current_triplets = [None for _ in range(channel.n_clients)]
        self.other_paras = [None for _ in range(channel.n_clients)]

        self.shared_own_mat = [None for _ in range(channel.n_clients)]
        self.shared_other_mat = [None for _ in range(channel.n_clients)]

        self.recovered_own_value = [None for _ in range(channel.n_clients)]
        self.recovered_other_value = [None for _ in range(channel.n_clients)]
        self.shared_out_AB = [None for _ in range(channel.n_clients)]
        self.shared_out_BA = [None for _ in range(channel.n_clients)]

        self.calc_threads = [None for _ in range(channel.n_clients)]
        self.working = True


    def __calculate_first_hidden_layer(self, other_id):
        """
        :param other_id:
        :return:
        """
        def get_next_batch():
            self.batch_data = np.random.uniform(-1, 1, [self.batch_size, self.data_dim])

        def send_data_dim():
            self.send_check_msg(other_id, ComputationMessage(MessageType.DATA_DIM, self.data_dim))

        def get_data_dim():
            msg = self.receive_check_msg(other_id, MessageType.DATA_DIM)
            self.other_paras[other_id] = np.random.normal(0, 1, [msg.data, self.output_dim])

        # 提供数据作为矩阵乘法中的乘数
        def set_triplet_AB():
            self.send_check_msg(self.triplets_id,
                          ComputationMessage(MessageType.SET_TRIPLET, (other_id, self.batch_data.shape,
                                                                       self.para.shape)))

        # 提供参数作为矩阵乘法中的的被乘数
        def set_triplet_BA():
            self.send_check_msg(self.triplets_id,
                          ComputationMessage(MessageType.SET_TRIPLET, (other_id, self.other_paras[other_id].shape,
                                                                       (self.batch_size, self.other_paras[other_id].shape[0]))))

        def get_triples():
            msg = self.receive_check_msg(self.triplets_id, MessageType.TRIPLE_ARRAY)
            self.current_triplets[msg.data[0]] = msg.data[1:]

        def share_data():
            self.shared_own_mat[other_id] = self.batch_data * np.random.uniform(0, 1, self.batch_data.shape)
            self.send_check_msg(other_id, ComputationMessage(MessageType.MUL_DATA_SHARE, self.batch_data - self.shared_own_mat[other_id]))

        def share_para():
            self.shared_own_mat[other_id] = self.other_paras[other_id] * \
                                            np.random.uniform(0, 1, self.other_paras[other_id].shape)
            self.send_check_msg(other_id, ComputationMessage(MessageType.MUL_DATA_SHARE, self.other_paras[other_id] - self.shared_own_mat[other_id]))

        def get_other_share():
            other_share = self.receive_check_msg(other_id, MessageType.MUL_DATA_SHARE)
            self.shared_other_mat[other_id] = other_share.data

        def recover_own_value():
            self.send_check_msg(other_id, ComputationMessage(MessageType.MUL_OwnVal_SHARE,
                                                       self.shared_own_mat[other_id] - self.current_triplets[other_id][0]))

        def get_other_value_share():
            msg = self.receive_check_msg(other_id, MessageType.MUL_OwnVal_SHARE)
            self.recovered_other_value[other_id] = self.shared_other_mat[other_id] - self.current_triplets[other_id][1] + msg.data

        def recover_other_value():
            self.send_check_msg(other_id, ComputationMessage(MessageType.MUL_OtherVal_SHARE,
                                                       self.shared_other_mat[other_id] - self.current_triplets[other_id][1]))

        def get_own_value_share():
            msg = self.receive_check_msg(other_id, MessageType.MUL_OtherVal_SHARE)
            self.recovered_own_value[other_id] = self.shared_own_mat[other_id] - self.current_triplets[other_id][0] + msg.data

        def get_shared_out_AB():
            self.shared_out_AB[other_id] = - np.matmul(self.recovered_own_value[other_id],
                                                       self.recovered_other_value[other_id])
            self.shared_out_AB[other_id] += np.matmul(self.shared_own_mat[other_id], self.recovered_other_value[other_id]) + \
                                            np.matmul(self.recovered_own_value[other_id], self.shared_other_mat[other_id]) + self.current_triplets[other_id][2]

        def get_shared_out_BA():
            self.shared_out_BA[other_id] = np.matmul(self.recovered_other_value[other_id], self.shared_own_mat[other_id]) + \
                                           np.matmul(self.shared_other_mat[other_id], self.recovered_own_value[other_id]) + self.current_triplets[other_id][2]

        # Calculate X_own * Theta_other
        def calc_AB():
            set_triplet_AB()
            get_triples()
            share_data()
            get_other_share()
            recover_own_value()
            get_other_value_share()
            recover_other_value()
            get_own_value_share()
            get_shared_out_AB()

        # Calculate Theta_own * X_other
        def calc_BA():
            set_triplet_BA()
            get_triples()
            share_para()
            get_other_share()
            recover_own_value()
            get_other_value_share()
            recover_other_value()
            get_own_value_share()
            get_shared_out_BA()
        try:
            get_next_batch()
            send_data_dim()
            get_data_dim()

            if other_id < self.client_id:
                calc_AB()
                calc_BA()
            else:
                calc_BA()
                calc_AB()
        except Exception as e:
            self.logger.logE("Python Exception: \n" + traceback.format_exc())
        finally:
            return

    def start_calc_first_layer(self, other_id):
        self.working = True
        thread = threading.Thread(target=self.__calculate_first_hidden_layer, args=(other_id,))
        self.calc_threads[other_id] = thread
        thread.start()
        return thread

    def stop_work(self):
        self.working = False


class LabelClient(BaseClient):
    def __init__(self, client_id: int, channel: BaseChannel, label_loader: DataLoader, batch_size: int, label_dim: int, server_id):
        super(LabelClient, self).__init__(client_id, channel)
        self.label_loader = label_loader
        self.batch_size = batch_size
        self.label_dim = label_dim
        self.server_id = server_id