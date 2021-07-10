import enum
import torch
import torch.nn as nn

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


class InputModel():
    """ Abstract base class for all implemented InputModel. """

    def get_layer_names(self):
        raise NotImplementedError()

    def create_input_model(self, c1, is_lstm, c2=None):
        raise NotImplementedError()

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, *x.shape[2:])  # (samples * timesteps, input_size) = (samples * timesteps, C,H,W)

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
