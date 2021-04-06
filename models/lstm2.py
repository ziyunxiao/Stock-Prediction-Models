import sys
import warnings

import tensorflow as tf

# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()
import keras
from tensorflow.python.framework import ops
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm


import logging

# import logging.config
# logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)
tf.compat.v1.disable_eager_execution()


class Lstm:
    def __init__(
        self,
        learning_rate,
        num_layers,
        size,
        size_layer,
        output_size,
        forget_bias=0.1,
    ):
        def lstm_cell(size_layer):
            return tf.keras.layers.LSTMCell(size_layer)

        rnn_cells = tf.keras.layers.StackedRNNCells(
            [lstm_cell(size_layer) for _ in range(num_layers)]
        )
        self.X = tf.compat.v1.placeholder(tf.float32, (None, None, size))
        self.Y = tf.compat.v1.placeholder(tf.float32, (None, output_size))
        drop = tf.compat.v1.nn.rnn_cell.DropoutWrapper(
            rnn_cells, output_keep_prob=forget_bias
        )
        self.hidden_layer = tf.compat.v1.placeholder(
            tf.float32, (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = keras.layers.RNN(
            drop, self.X, initial_state=self.hidden_layer, dtype=tf.float32
        )
        self.logits = tf.compat.v1.layers.dense(self.outputs[-1], output_size)
        self.cost = tf.reduce_mean(input_tensor=tf.square(self.Y - self.logits))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )


class MLUtil:
    def __init__(self) -> None:
        sns.set()
        tf.compat.v1.random.set_random_seed(1234)
        # if not sys.warnoptions:
        #     warnings.simplefilter('ignore')

        # default settings
        self.test_size = 30
        self.simulation_size = 10

    def load_data(self, filename) -> None:
        df = pd.read_csv(filename)
        df.head()

        minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype("float32"))  # Close index
        df_log = minmax.transform(df.iloc[:, 4:5].astype("float32"))  # Close index
        df_log = pd.DataFrame(df_log)
        logger.info("Dataframe sample")
        logger.info(df_log.head())

        self.df = df
        self.df_log = df_log
        self.minmax = minmax

    def split_dataset(self, test_size=30, simulation_size=10):
        self.test_size = test_size
        self.simulation_size = simulation_size

        self.df_train = self.df_log.iloc[:-test_size]
        self.df_test = self.df_log.iloc[-test_size:]
        logger.info(self.df.shape, self.df_train.shape, self.df_test.shape)

    def calculate_accuracy(self, real, predict):
        real = np.array(real) + 1
        predict = np.array(predict) + 1
        percentage = 1 - np.sqrt(np.mean(np.square((real - predict) / real)))
        return percentage * 100

    def anchor(self, signal, weight):
        buffer = []
        last = signal[0]
        for i in signal:
            smoothed_val = last * weight + (1 - weight) * i
            buffer.append(smoothed_val)
            last = smoothed_val
        return buffer

    def train_lstm(self):

        num_layers = 1
        size_layer = 128
        timestamp = 5
        epoch = 300
        dropout_rate = 0.8
        future_day = self.test_size
        learning_rate = 0.01
        modelnn = Lstm(
            learning_rate,
            num_layers,
            self.df_log.shape[1],
            size_layer,
            self.df_log.shape[1],
            dropout_rate,
        )

        df = self.df
        df_train = self.df_train
        test_size = self.test_size
        minmax = self.minmax

        ops.reset_default_graph()
        sess = tf.compat.v1.InteractiveSession()
        sess.run(tf.compat.v1.global_variables_initializer())
        date_ori = pd.to_datetime(df.iloc[:, 0]).tolist()

        pbar = tqdm(range(epoch), desc="train loop")
        for i in pbar:
            init_value = np.zeros((1, num_layers * 2 * size_layer))
            total_loss, total_acc = [], []
            for k in range(0, df_train.shape[0] - 1, timestamp):
                index = min(k + timestamp, df_train.shape[0] - 1)
                batch_x = np.expand_dims(df_train.iloc[k:index, :].values, axis=0)
                batch_y = df_train.iloc[k + 1 : index + 1, :].values
                logits, last_state, _, loss = sess.run(
                    [
                        modelnn.logits,
                        modelnn.last_state,
                        modelnn.optimizer,
                        modelnn.cost,
                    ],
                    feed_dict={
                        modelnn.X: batch_x,
                        modelnn.Y: batch_y,
                        modelnn.hidden_layer: init_value,
                    },
                )
                init_value = last_state
                total_loss.append(loss)
                total_acc.append(self.calculate_accuracy(batch_y[:, 0], logits[:, 0]))
            pbar.set_postfix(cost=np.mean(total_loss), acc=np.mean(total_acc))

        future_day = test_size

        output_predict = np.zeros((df_train.shape[0] + future_day, df_train.shape[1]))
        output_predict[0] = df_train.iloc[0]
        upper_b = (df_train.shape[0] // timestamp) * timestamp
        init_value = np.zeros((1, num_layers * 2 * size_layer))

        for k in range(0, (df_train.shape[0] // timestamp) * timestamp, timestamp):
            out_logits, last_state = sess.run(
                [modelnn.logits, modelnn.last_state],
                feed_dict={
                    modelnn.X: np.expand_dims(df_train.iloc[k : k + timestamp], axis=0),
                    modelnn.hidden_layer: init_value,
                },
            )
            init_value = last_state
            output_predict[k + 1 : k + timestamp + 1] = out_logits

        if upper_b != df_train.shape[0]:
            out_logits, last_state = sess.run(
                [modelnn.logits, modelnn.last_state],
                feed_dict={
                    modelnn.X: np.expand_dims(df_train.iloc[upper_b:], axis=0),
                    modelnn.hidden_layer: init_value,
                },
            )
            output_predict[upper_b + 1 : df_train.shape[0] + 1] = out_logits
            future_day -= 1
            date_ori.append(date_ori[-1] + timedelta(days=1))

        init_value = last_state

        for i in range(future_day):
            o = output_predict[-future_day - timestamp + i : -future_day + i]
            out_logits, last_state = sess.run(
                [modelnn.logits, modelnn.last_state],
                feed_dict={
                    modelnn.X: np.expand_dims(o, axis=0),
                    modelnn.hidden_layer: init_value,
                },
            )
            init_value = last_state
            output_predict[-future_day + i] = out_logits[-1]
            date_ori.append(date_ori[-1] + timedelta(days=1))

        output_predict = minmax.inverse_transform(output_predict)
        deep_future = self.anchor(output_predict[:, 0], 0.3)

        return deep_future[-test_size:]

    def test_lstm(self):
        self.load_data("./dataset/GOOG-year.csv")
        results = []
        for i in range(self.simulation_size):
            logger.info("simulation %d" % (i + 1))
            results.append(self.train_lstm())


if __name__ == "__main__":
    inst = MLUtil()
    inst.test_lstm()
