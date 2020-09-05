import numpy as np
import threading
import Client.MPCClient as MPCC
from Communication.Message import MessageType, PackedMessage
from Communication.Channel import BaseChannel
from Client.Data.DataLoader import DataLoader
from Client.Common.BroadcastClient import BroadcastClient
from Utils.Log import Logger
from Client.SecureXGBoost.xgb_scratch import XGBClassifier, CART, LogLoss


class DataClient(MPCC.DataClient):
    def __init__(self, channel: BaseChannel, logger: Logger,
                 mpc_paras: MPCC.MPCClientParas,
                 data_loader: DataLoader, test_data_loader: DataLoader):
        super(DataClient, self).__init__(channel, logger, mpc_paras,
                                         data_loader, test_data_loader)
        # 将当前client id从feature clients中移除
        self.other_feature_client_ids = self.feature_client_ids.copy()
        if self.client_id in self.other_feature_client_ids:
            self.other_feature_client_ids.remove(self.client_id)
        # init broadcast client
        self.broadcaster = BroadcastClient(self.channel, self.logger)

        # 超参数信息
        self.configs = None
        self.learning_rate = 0
        self.max_iteration = 0

class FeatureClient(DataClient):
    def __init__(self, channel: BaseChannel, logger: Logger,
                 mpc_paras: MPCC.MPCClientParas,
                 data_loader: DataLoader, test_data_loader: DataLoader):
        # random generate some to data
        super(FeatureClient, self).__init__(channel, logger, mpc_paras, data_loader, test_data_loader)

        self.batch_data = None
        self.para = None
        self.max_depth = 0
        self.learning_rate = None
        self.reg_lambda = 0
        self.gamma = 0
        self.col_sample_ratio = 0
        self.row_sample_ratio = 0

        self.error = False
        self.finished = False

        # get data
        self.data = None
        self.test_data = None
        self.data = self.train_data_loader.get_batch(None)
        self.test_data = self.test_data_loader.get_batch(None)

        self.raw_label = None
        self.estimators = []

        self.logger.log("Client initialized")

    def set_config(self, config: dict):
        """配置训练参数，config是从server得到的dict"""
        self.configs = config
        self.learning_rate = config["learning_rate"]
        self.max_iteration = config["max_iteration"]
        self.max_depth = config["max_depth"]
        self.reg_lambda = config["reg_lambda"]
        self.gamma = config["gamma"]
        self.col_sample_ratio = config["col_sample_ratio"]
        self.row_sample_ratio = config["row_sample_ratio"]

    def _before_trainning(self, wait_for_server: float = 100):
        """
        Receive config message from server, then initialize some parameters
        After this, send CLIENT_READY message to server
        """
        self.logger.log("Client started, waiting for server config message with time out %.2f" % wait_for_server)
        try:
            msg = self.receive_check_msg(self.main_client_id, MessageType.XGBOOST_TRAIN_CONFIG, time_out=wait_for_server)
            self.set_config(msg.data)
            # self.send_check_msg(self.main_client_id, PackedMessage(MessageType.CLIENT_READY, None))
        except:
            self.logger.logE("Python Exception encountered, stop.")
            self.logger.logE("Train not started")
            return False

        self.logger.log("Received train conifg message: %s" % msg.data)
        return True

    def start_train(self, wait_for_server: float = 100):
        """
        Receive config message from server, then initialize some parameters
        After this, send CLIENT_READY message to server
        """
        if not self._before_trainning():
            return False

        print("data client start train")
        # start train
        n_rounds = 0
        # for i in range(self.max_iteration):
        while True:
            train_res = self._train_one_round()
            n_rounds += 1
            """
            After one train round over, send CLIENT_ROUND_OVER message to server
            """
            # try:
            #     self.send_check_msg(self.main_client_id, PackedMessage(MessageType.CLIENT_ROUND_OVER, train_res))
            # except:
            #     self.logger.logE("Error encountered while sending round over message to server")
            #     break
            if not train_res:
                self.logger.logE("Error encountered while training one round. Stop.")
                return False
            if self.finished:
                return True
            self.logger.log("Data Client %d: Train round %d finished" % (self.client_id, n_rounds))

    def _local_train_one_round(self):
        """本地调用xgb计算一轮，发送gain信息"""

        # todo 本地调用xgb计算一轮，发送gain信息

        # send gain
        def send_gain_to(client_id: int, update: float):
            try:
                self.send_check_msg(client_id, PackedMessage(MessageType.XGBOOST_GAIN, update))
            except:
                self.logger.logE("Error encountered while sending gain to other client")
                self.error = True

        # get label
        # label_data = None
        # def get_label(client_id: int):
        #     try:
        #         nonlocal label_data
        #         label_data=self.receive_check_msg(client_id, MessageType.NEXT_TRAIN_ROUND).data
        #         print(label_data)
        #     except:
        #         self.logger.logE("Error encountered while receiving label from other client")
        #         self.error = True
        #
        # receving_th = threading.Thread(target=get_label, args=(self.main_client_id,))
        # receving_th.start()
        # receving_th.join()

        # send label
        def send_label(client_id: int, label: np.ndarray):
            try:
                self.send_check_msg(client_id, PackedMessage(MessageType.XGBOOST_RESIDUAL, label))
            except:
                self.logger.logE("Error encountered while sending label to other client")
                self.error = True

        # recevie select_node
        select_node = None

        def receive_node(client_id: int):
            try:
                nonlocal select_node
                select_node = self.receive_check_msg(client_id, MessageType.XGBOOST_SELECTED_NODE).data
            except:
                self.logger.logE("Error encountered while receiving selected info from client %d" % client_id)
                self.error = True

        X_train = self.data
        X_test = self.test_data
        y_train = self.label_data
        y = self.raw_label

        # self.mean = np.mean(y)
        # y_pred = np.ones_like(y) * self.mean

        # calculate loss
        if self.raw_label.all() == self.label_data.all():
            loss = LogLoss(self.raw_label, np.ones_like(self.raw_label) * np.mean(self.raw_label))
        else:
            loss = LogLoss(self.raw_label, self.raw_label - self.label_data)
        g, h = loss.g(), loss.h()

        # print('h:', h)
        estimator_t = CART(reg_lambda=self.reg_lambda, max_depth=self.max_depth, gamma=self.gamma,
                           col_sample_ratio=self.col_sample_ratio, row_sample_ratio=self.row_sample_ratio)
        estimator_t.fit(X_train, self.label_data, g, h)
        # print(y_pred.shape)
        # print(np.expand_dims(estimator_t.predict(X_train), axis=1).shape)
        label = self.label_data - (self.learning_rate * np.expand_dims(estimator_t.predict(X_train), axis=1))
        # print(y_pred)
        # the smaller the better
        # print(estimator_t.obj_val)

        # tinyxgb_clf = XGBClassifier(self.configs)
        # print(y_train)

        # tinyxgb_clf.fit(X_train, y_train)
        gain = estimator_t.obj_val
        sending_th = threading.Thread(target=send_gain_to, args=(self.main_client_id, gain,))
        sending_th.start()
        sending_th.join()

        receving_th = threading.Thread(target=receive_node, args=(self.main_client_id,))
        receving_th.start()
        receving_th.join()

        # label = y_train - y_pred
        # print('{} selected: {}'.format(self.client_id, select_node))

        if select_node == 'ack':
            print('epoch ---- loss', np.mean(loss.forward()), '----client_id', self.client_id)
            self.estimators.append(estimator_t)
            self.send_check_msg(self.main_client_id, PackedMessage(MessageType.XGBOOST_RESIDUAL, label))
            # sending_th = threading.Thread(target=send_label, args=(self.main_client_id, label))
            # sending_th.start()
            # sending_th.join()

    def _train_one_round(self):
        """
        :return: `True` if no error occurred during this training round. `False` otherwise
        """

        def send_pred(server_id, update):
            try:
                self.send_check_msg(server_id, PackedMessage(MessageType.XGBOOST_PRED_LABEL, update))
            except:
                self.logger.logE("Error encountered while sending label to other client")
                self.error = True

        try:
            """Waiting for server's next-round message"""
            start_msg = self.receive_check_msg(self.main_client_id,
                                               [MessageType.XGBOOST_NEXT_TRAIN_ROUND, MessageType.XGBOOST_TRAINING_STOP])
            """
            If server's message is stop message, stop
            otherwise, if server's next-round message's data is "Test", switch to test mode
            """
            # print(start_msg.header)
            if start_msg.header == MessageType.XGBOOST_TRAINING_STOP:
                self.logger.log("Received server's training stop message, stop training")

                y_pred = 0
                for estimator in self.estimators:
                    y_pred += estimator.predict(self.test_data)
                print(y_pred)
                sending_th = threading.Thread(target=send_pred, args=(self.main_client_id, y_pred,))
                sending_th.start()
                sending_th.join()
                self.finished = True
                return True
        except:
            self.logger.logE("Error encountered while receiving server's start message")
            return False

        self.label_data = start_msg.data
        if self.raw_label is None:
            self.raw_label = start_msg.data

        self._local_train_one_round()

        return True

class LabelClient(DataClient):
    def __init__(self, channel: BaseChannel, logger: Logger,
                 mpc_paras: MPCC.MPCClientParas,
                 label_loader: DataLoader, test_label_loader: DataLoader):
        super(LabelClient, self).__init__(channel, logger, mpc_paras, label_loader, test_label_loader)
        self.finish = False
        self.logger.log("LabelClient has initialized...")

    def _before_training(self):
        try:
            config = self.receive_check_msg(self.main_client_id, MessageType.XGBOOST_TRAIN_CONFIG).data
            self.logger.log("Received main client's config message: {}".format(config))
            self.batch_size = config["batch_size"]
            self.test_batch_size = config["test_batch_size"]
        except:
            self.logger.logE("Get training config from server failed. Stop training.")
            return False
        # send label to main client
        if not self.send_label_to_main():
            return False
        return True

    def send_label_to_main(self):
        try:
            header = MessageType.XGBOOST_LABEL

            train = self.receive_check_msg(self.main_client_id, MessageType.XGBOOST_TRAIN).data
            if train:
                data = self.train_data_loader.get_batch(self.batch_size)
            else:
                data = self.test_data_loader.get_batch(self.test_batch_size)
            self.send_check_msg(self.main_client_id,
                                PackedMessage(header, data))
        except:
            self.logger.logE("Send Label to main client predictions failed. Stop training.")
            return False

        return True

    def _train_one_round(self):
        try:
            rev = self.receive_check_msg(self.main_client_id,
                                         [MessageType.XGBOOST_NEXT_TRAIN_ROUND, MessageType.XGBOOST_TRAINING_STOP])
            if rev.header == MessageType.XGBOOST_TRAINING_STOP:
                self.finish = True
        except:
            self.logger.logE("Error encountered while receiving server's start message")
            return False
        return True

    def start_train(self):
        self.logger.log("LabelClient(Server) started")
        if not self._before_training():
            return False

        while True:
            if not self._train_one_round():
                return False
            if self.finish:
                break
        # send test label to main
        if not self.send_label_to_main():
            return False
        return True

