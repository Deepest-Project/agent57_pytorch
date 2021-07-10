import enum

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# from keras import backend as K
# from keras.layers import Input, Flatten, Permute, LSTM, Dense, Concatenate, Reshape, Lambda, Conv2D, \
#     MaxPooling2D, Activation, Add
# from keras.models import Model


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, *x.shape[
                                             2:])  # (samples * timesteps, input_size) = (samples * timesteps, C,H,W)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, *y.shape[1:])  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), *y.shape[1:])  # (timesteps, samples, output_size)

        return y


class ValueModel(InputModel):
    def __init__(self, input_unit, dense_units, layer_num=3):
        self.input_unit = input_unit
        self.dense_units = dense_units
        self.layer_num = layer_num

    def get_layer_names(self):
        return ["l{}".format(i) for i in range(self.layer_num)]

    def create_input_model(self, c1, is_lstm, c2=None):

        if not is_lstm:
            prev = self.input_unit
            for i in range(self.layer_num):
                l = nn.Linear(prev, self.dense_units[i], )
                prev = self.dense_units[i]
                c1 = l(c1)
                if c2 is not None:
                    c2 = l(c2)
        else:
            prev = self.input_unit
            for i in range(self.layer_num):
                l = TimeDistributed(nn.Linear(prev, self.dense_units[i]))
                prev = self.dense_units[i]
                c1 = l(c1)
                if c2 is not None:
                    c2 = l(c2)

        return c1, c2


def clipped_error_loss(y_true, y_pred):
    err = y_true - y_pred  # エラー
    L2 = 0.5 * K.square(err)
    L1 = K.abs(err) - 0.5

    # エラーが[-1,1]区間ならL2、それ以外ならL1を選択する。
    loss = tf.where((K.abs(err) < 1.0), L2, L1)  # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)


class InputType(enum.Enum):
    VALUES = 1  # 画像無し
    GRAY_2ch = 3  # (width, height)
    GRAY_3ch = 4  # (width, height, 1)
    COLOR = 5  # (width, height, ch)


class DuelingNetwork(enum.Enum):
    AVERAGE = 0
    MAX = 1
    NAIVE = 2


class LstmType(enum.Enum):
    NONE = 0
    STATELESS = 1
    STATEFUL = 2


class UvfaType(enum.Enum):
    ACTION = 1
    REWARD_EXT = 2
    REWARD_INT = 3
    POLICY = 4


from collections import namedtuple

ModelBuilder = namedtuple("ModelBuilder",
                          ["input_shape", "input_type", "input_model", "input_model_emb", "input_model_rnd",
                           "batch_size", "nb_actions", "input_sequence", "enable_dueling_network",
                           "dueling_network_type", "dense_units_num", "lstm_type", "lstm_units_num", "policy_num"])


class actval_func_model(nn.Module):
    def __init__(self, optimizer, uvfa):
        # self.uvfa = uvfa
        super().__init__()
        uvfa_input_num = 0
        if UvfaType.ACTION in uvfa:
            uvfa_input_num += self.nb_actions
        if UvfaType.REWARD_EXT in uvfa:
            uvfa_input_num += 1
        if UvfaType.REWARD_INT in uvfa:
            uvfa_input_num += 1
        if UvfaType.POLICY in uvfa:
            uvfa_input_num += self.policy_num

        if len(uvfa) > 0:
            self.lstm = nn.LSTM(input_size=self.input_shape + 1, hidden_size=self.lstm_units_num, batch_first=True)
        else:
            self.lstm = nn.LSTM(input_size=self.input_shape, hidden_size=self.lstm_units_num, batch_first=True)

        self.fc1 = nn.Linear(self.lstm_units_num, self.dense_units_num)
        self.fc2 = nn.Linear(self.dense_units_num, self.nb_actions)
        self.optimizer = optimizer(self.parameters())

    def forward(self, x):
        x, _ = self.lstm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def train_on_batch(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x)
        loss = nn.MSELoss()(out, y)
        loss.backward()
        self.optimizer.step()


class embedding_model(nn.module):
    def __init__(self, builder):
        super().__init__()
        self.fc1 = nn.Linear(builder.input_shape, 32)

    def forward(self, x):
        # x = self._input_layer(x)
        # x = self._image_layer(x)
        x = F.relu(self.fc1(x))

        return x


class embedding_model_classifier(nn.Module):
    def __init__(self, builder, loss=nn.CrossEntropyLoss, optimizer=optim.Adam):
        super().__init__()
        self.fc1 = nn.Linear(builder.input_shape, 32)
        self.fc2 = nn.Linear(32, 128)
        self.classfier = nn.Softmax()

        self.loss = loss()
        self.optimizer = optimizer(self.parameters())

    def forward(self, c1, c2):
        c1 = F.relu(self.fc1(c1))
        c2 = F.relu(self.fc1(c2))

        x = F.relu(self.fc2(torch.cat([c1, c2])))
        x = self.classfier(x)

        return x

    def train_on_batch(self, c1, c2, action):
        x = self.forward(c1, c2)
        self.optimizer.zero_grad()
        loss = self.loss(x, action)
        loss.backward()
        self.optimizer.step()


class rnd_model(nn.Module): # todo : instance 시에 optimizer ㄴ허ㅓㅇ쥑
    def __init__(self, builder, optimizer):
        super().__init__()
        self.fc1 = nn.Linear(builder.input_shape, 128)
        self.optimizer = optimizer(self.parameters())

    def forward(self, x):
        self.fc1(x)
        return x

    def train_on_batch(self, x, y):
        self.optimizer.zero_grad()
        out = self.forward(x)
        loss = nn.MSELoss()(out, y)
        loss.backward()
        self.optimizer.step()


class InputModel():
    """ Abstract base class for all implemented InputModel. """

    def get_layer_names(self):
        raise NotImplementedError()

    def create_input_model(self, c1, is_lstm, c2=None):
        raise NotImplementedError()


class DQNImageModel(InputModel):
    """ native dqn image model
    https://arxiv.org/abs/1312.5602
    """

    def get_layer_names(self):
        return [
            "c1",
            "c2",
            "c3",
        ]

    def create_input_model(self, c1, is_lstm, c2=None):

        if not is_lstm:
            l = Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation="relu", name="c1")
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)

            l = Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu", name="c2")
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)

            l = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu", name="c3")
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)

            c1 = Flatten()(c1)
            if c2 is not None:
                c2 = Flatten()(c2)

        else:
            l = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation="relu"), name="c1")
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)

            l = TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu"), name="c2")
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)

            l = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu"), name="c3")
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)

            c1 = TimeDistributed(Flatten())(c1)
            if c2 is not None:
                c2 = TimeDistributed(Flatten())(c2)

        return c1, c2


class R2D3ImageModel(InputModel):
    """ R2D3 image model
    https://arxiv.org/abs/1909.01387
    """

    def __init__(self):
        self.names = []

    def get_layer_names(self):
        return self.names

    def create_input_model(self, c1, is_lstm, c2=None):
        self.names = []

        c1, c2 = self._resblock(c1, 16, is_lstm, c2)
        c1, c2 = self._resblock(c1, 32, is_lstm, c2)
        c1, c2 = self._resblock(c1, 32, is_lstm, c2)

        c1 = Activation("relu")(c1)
        if c2 is not None:
            c2 = Activation("relu")(c2)

        if not is_lstm:
            c1 = Flatten()(c1)
            if c2 is not None:
                c2 = Flatten()(c2)
        else:
            c1 = TimeDistributed(Flatten())(c1)
            if c2 is not None:
                c2 = TimeDistributed(Flatten())(c2)

        return c1, c2

    def _resblock(self, c1, n_filter, is_lstm, c2):
        if not is_lstm:
            n = "l{}".format(len(self.names))
            self.names.append(n)
            l = Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same", name=n)
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)
            l = MaxPooling2D((3, 3), strides=(2, 2), padding='same')
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)
        else:
            n = "l{}".format(len(self.names))
            self.names.append(n)
            l = TimeDistributed(Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same"), name=n)
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)
            l = TimeDistributed(MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)

        c1, c2 = self._residual_block(c1, n_filter, is_lstm, c2)
        c1, c2 = self._residual_block(c1, n_filter, is_lstm, c2)

        return c1, c2

    def _residual_block(self, c1, n_filter, is_lstm, c2):

        if not is_lstm:
            n = "l{}".format(len(self.names))
            self.names.append(n)
            l = Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same", name=n)
            c1_tmp = Activation("relu")(c1)
            c1_tmp = l(c1_tmp)
            if c2 is not None:
                c2_tmp = Activation("relu")(c2)
                c2_tmp = l(c2_tmp)

            n = "l{}".format(len(self.names))
            self.names.append(n)
            l = Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same", name=n)
            c1_tmp = Activation("relu")(c1_tmp)
            c1_tmp = l(c1_tmp)
            if c2 is not None:
                c2_tmp = Activation("relu")(c2)
                c2_tmp = l(c2_tmp)
        else:
            n = "l{}".format(len(self.names))
            self.names.append(n)
            l = TimeDistributed(Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same"), name=n)
            c1_tmp = Activation("relu")(c1)
            c1_tmp = l(c1_tmp)
            if c2 is not None:
                c2_tmp = Activation("relu")(c2)
                c2_tmp = l(c2_tmp)

            n = "l{}".format(len(self.names))
            self.names.append(n)
            l = TimeDistributed(Conv2D(n_filter, (3, 3), strides=(1, 1), padding="same"), name=n)
            c1_tmp = Activation("relu")(c1_tmp)
            c1_tmp = l(c1_tmp)
            if c2 is not None:
                c2_tmp = Activation("relu")(c2)
                c2_tmp = l(c2_tmp)

        # 結合
        c1 = Add()([c1, c1_tmp])
        if c2 is not None:
            c2 = Add()([c2, c2_tmp])
        return c1, c2

# class ValueModel(InputModel):
#     def __init__(self, dense_units, layer_num=3):
#         self.dense_units = dense_units
#         self.layer_num = layer_num
#
#     def get_layer_names(self):
#         return ["l{}".format(i) for i in range(self.layer_num)]
#
#     def create_input_model(self, c1, is_lstm, c2=None):
#
#         if not is_lstm:
#             for i in range(self.layer_num):
#                 l = Dense(self.dense_units, activation="relu", name="l{}".format(i))
#                 c1 = l(c1)
#                 if c2 is not None:
#                     c2 = l(c2)
#         else:
#             for i in range(self.layer_num):
#                 l = TimeDistributed(Dense(self.dense_units, activation="relu", name="l{}".format(i)))
#                 c1 = l(c1)
#                 if c2 is not None:
#                     c2 = l(c2)
#
#         return c1, c2
