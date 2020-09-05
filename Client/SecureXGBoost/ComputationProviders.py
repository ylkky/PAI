import threading
import numpy as np
import traceback
from Communication.Message import PackedMessage, MessageType
from Communication.Channel import BaseChannel
import Client.MPCClient as MPCC
from Client.Common.BroadcastClient import BroadcastClient
from Utils.Log import Logger
from Client.Learning.Metrics import onehot_accuracy

class MainClient(MPCC.MainClient):
    def __init__(self, channel: BaseChannel, logger: Logger,
                 mpc_params: MPCC.MPCClientParas, metric_func,
                 config):
        super(MainClient, self).__init__(channel, logger, mpc_params)
        self.config = config

        self.error = False
        self.finished = False

        if metric_func is None:
            self.metric_func = onehot_accuracy
        else:
            self.metric_func = metric_func

        self.broadcaster = BroadcastClient(self.channel, self.logger)
        # get labels
        self.label_data = None

    def _before_training(self):
        # send config dict to every data client
        self.broadcaster.broadcast(self.feature_client_ids + [self.label_client_id],
                                   PackedMessage(MessageType.XGBOOST_TRAIN_CONFIG, self.config
        ))
        if self.broadcaster.error:
            self.logger.logE("Broadcast training config message failed. Stop training.")
            return False

        if not self._get_raw_abel_data():
            return False
        # self.broadcaster.receive_all(self.feature_client_ids, MessageType.CLIENT_READY)
        return True

    def _get_raw_abel_data(self):
        try:
            self.send_check_msg(self.label_client_id,
                                PackedMessage(MessageType.XGBOOST_TRAIN, True))
            self.label_data = self.receive_check_msg(self.label_client_id, MessageType.XGBOOST_LABEL).data
        except:
            self.logger.logE("Receive label from label client failed. Stop training.")
            return False
        return True

    def get_residual(self, client):
        """
        从selected client处获取新的label，赋值给self.label_data
        """
        try:
            self.label_data = self.receive_check_msg(client, MessageType.XGBOOST_RESIDUAL).data
        except:
            self.logger.logE("Receive residual loss from selected client failed. Stop training.")
            return False
        return True


    def compare_split_node(self):
        client_outs = self.broadcaster.receive_all(self.feature_client_ids, MessageType.XGBOOST_GAIN)
        if self.broadcaster.error:
            self.logger.logE("Gather clients' outputs failed. Stop training")
            return False
        # send ack msg to the client which has the min loss
        selected = None
        for data_client in self.feature_client_ids:
            if selected is None or client_outs[selected] > client_outs[data_client]:
                selected = data_client

        msgs = dict()
        for data_client in self.feature_client_ids:
            if data_client == selected:
                msgs[data_client] = PackedMessage(MessageType.XGBOOST_SELECTED_NODE, 'ack')
            else:
                msgs[data_client] = PackedMessage(MessageType.XGBOOST_SELECTED_NODE, 'rej')

        self.broadcaster.broadcast(self.feature_client_ids, msgs)
        if self.broadcaster.error:
            self.logger.logE("Broadcast split info failed. Stop training")
            return False

        if not self.get_residual(selected):
            return False

        return True

    def start_train(self):
        if not self._before_training():
            return False

        self.logger.log("MainClient started")
        n_rounds = 0
        for i in range(self.config['max_iteration']):  # server控制最大迭代次数
            train_res = self._train_one_round()
            n_rounds += 1
            self.logger.log("MainClient: Train round %d finished" % n_rounds)
            # if n_rounds > 5:
            #     self._broadcast_start(stop=True)
            #     break
            if not train_res:
                self.logger.logE("Training stopped due to error")
                self._broadcast_start(stop=True)
                return False
        self._broadcast_start(stop=True)

        self.predict()
        return True

    def _train_one_round(self):
        """Broadcast start message to every client"""
        self._broadcast_start()
        if self.broadcaster.error:
            self.logger.logE("Error encountered while broadcasting start messages")
            return False

        """get gain information and compare split node"""
        train_res = self.compare_split_node()
        if not train_res:
            self.logger.logE("Error encountered while splitting nodes")
            return False

        return True

    def _broadcast_start(self, stop=False):
        if not stop:
            header = MessageType.XGBOOST_NEXT_TRAIN_ROUND
        else:
            header = MessageType.XGBOOST_TRAINING_STOP
        start_data = self.label_data
        self.broadcaster.broadcast(self.feature_client_ids + [self.label_client_id], PackedMessage(header, start_data))

    def predict(self):
        self.logger.log("MainClient predict start...")

        # get test label
        try:
            self.send_check_msg(self.label_client_id,
                                PackedMessage(MessageType.XGBOOST_TRAIN, False))
            y_true = self.receive_check_msg(self.label_client_id, MessageType.XGBOOST_LABEL).data
        except:
            self.logger.logE("Get test label from label client failed. Stop predict.")
            return False

        y_preds = np.zeros((y_true.shape[0]))

        y_pred_dict = self.broadcaster.receive_all(self.feature_client_ids, MessageType.XGBOOST_PRED_LABEL)
        if self.broadcaster.error:
            self.logger.logE("Gather predict y failed")
            return False

        for y_pred in y_pred_dict.values():
            y_preds[:] += y_pred

        print(y_preds)
        res = self.metric_func(y_true, y_preds)
        print(res)
        self.logger.log("Predict auc={:.2f}, ks={:.2f}".format(*res))


