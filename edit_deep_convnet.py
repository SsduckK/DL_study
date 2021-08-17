# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from trainer_neuralnet import *
import pickle
from collections import OrderedDict
from edit_layer import *


class DeepConvNet:
    """정확도 99% 이상의 고정밀 합성곱 신경망

    네트워크 구성은 아래와 같음
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """

    def __init__(self, input_dim=(1, 28, 28), conv_list=[], hidden_size=50, output_size=10):
        # 가중치 초기화===========
        # 각 층의 뉴런 하나당 앞 층의 몇 개 뉴런과 연결되는가（TODO: 자동 계산되게 바꿀 것）
        self.conv_list = conv_list

        pre_node_nums = np.array(
            [1 * 3 * 3, 16 * 3 * 3, 16 * 3 * 3, 32 * 3 * 3, 32 * 3 * 3, 64 * 3 * 3, 64 * 4 * 4, hidden_size])
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값
        self.make_conv_params()
        self.cnt = 0

        self.conv_list[0] = {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}
        self.conv_list[1] = {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}
        self.conv_list[2] = {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1}
        self.conv_list[3] = {'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1}
        self.conv_list[4] = {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1}
        self.conv_list[5] = {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1}

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate(self.conv_list):
            self.params['W' + str(idx + 1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'],
                                                                                       pre_channel_num,
                                                                                       conv_param['filter_size'],
                                                                                       conv_param['filter_size'])
            self.params['b' + str(idx + 1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64 * 4 * 4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # 계층 생성===========
        self.layers = []
        for i in range(1, 4):
            self.conv_relu_Pooling(self.layers)
        self.layers.append(Affine(self.params['W' + str(self.cnt + 1)], self.params['b' + str(self.cnt + 1)]))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W' + str(self.cnt + 2)], self.params['b' + str(self.cnt + 2)]))
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithLoss()

    def make_conv_params(self, params=6):
        for list in range(0, params):
            self.conv_list.append(list)
        return self.conv_list

    def conv_relu_Pooling(self, layer):
        for i in range(1, 3):
            layer.append(Convolution(self.params['W' + str(self.cnt + 1)], self.params['b' + str(self.cnt + 1)],
                                     self.conv_list[self.cnt]['stride'], self.conv_list[self.cnt]['pad']))
            layer.append(Relu())
            self.cnt = self.cnt+1
        layer.append(Pooling(pool_h=2, pool_w=2, stride=2))
        return layer

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i + 1)] = self.layers[layer_idx].dW
            grads['b' + str(i + 1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i + 1)]
            self.layers[layer_idx].b = self.params['b' + str(i + 1)]


def main():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
    network = DeepConvNet()
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=20, mini_batch_size=100,
                      optimizer='Adam', optimizer_param={'lr':0.001},
                      evaluate_sample_num_per_epoch=1000)
    trainer.train()

    # 매개변수 보관
    network.save_params("deep_convnet_params.pkl")
    print("Saved Network Parameters!")


if __name__ == "__main__":
    main()