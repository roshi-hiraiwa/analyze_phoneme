# coding: utf_8
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from PIL import Image
import os

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


class A_classification:

    def __init__(self, cls_num):
        self.cls_num = cls_num
        self.input_shape = [-1, 40, 6, 1]
        self.all_loss_histories = []  # 全LOOのloss履歴を保存
        tf.keras.backend.set_floatx("float64")

    def set_model(self):
        # モデルの定義
        model = Sequential()
        model.add(
            Conv2D(
                32,
                3,
                input_shape=tuple(self.input_shape[1:]),
                padding="same",
                activation="relu",
            )
        )
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, 3, padding="same", activation="relu"))
        model.add(MaxPooling2D((2, 1)))
        model.add(Flatten())
        model.add(Dense(self.cls_num, activation="softmax"))
        model.compile(
            optimizer="adadelta", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        self.model = model

    def fit_model(self, x_train, y_train):
        # 学習
        self.history = self.model.fit(
            x_train, y_train, epochs=200, batch_size=30, shuffle=True, verbose=0
        )
        # loss履歴を保存
        self.all_loss_histories.append(self.history.history["loss"])

    def save(self, summary_txt, summary_png, model_name, loss_name):
        # モデルの構造の出力
        with open(summary_txt, "w") as fp:
            self.model.summary(print_fn=lambda x: fp.write(x + "\n"))

        # モデル図の保存
        plot_model(
            self.model,
            to_file=summary_png,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
        )

        # モデル周りの保存
        for i, loss_history in enumerate(self.all_loss_histories):
            plt.plot(loss_history)
        plt.title("model loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(loss_name)
        self.model.save(model_name)

    def prediction(self, x_test):
        # モデルへのtestデータの入力と結果の保存
        predict = self.model.predict(
            x_test.reshape(self.input_shape), verbose=0
        )  # 各クラスへの所属確率の予測を返す
        # predict = np.argmax(model.predict(x_test), axis = 1)  # 予測クラスを返す
        return predict

    def set_cams(self):
        os.makedirs("../target/img", exist_ok=True)
        self.class_length = self.model.output.shape[1]
        self.last_conv_layer_index = 2
        self.last_conv_layer_shape = self.model.layers[
            self.last_conv_layer_index
        ].output.shape
        self.cams = [
            np.empty([0, self.last_conv_layer_shape[1] * self.last_conv_layer_shape[2]])
            for s in range(self.class_length)
        ]
        self.xticks_label, self.xticks_value = [0, 39], [
            1,
            40,
        ]  # [0, 35, 75], [1, 36, 76]
        self.yticks_label, self.yticks_value = [0, 5], [6, 1]

    def set_grad_model(self):
        temp_file = "t.h5"
        self.model.save(temp_file)
        self.model = load_model(temp_file)
        self.model.layers[-1].activation = None
        self.grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.layers[self.last_conv_layer_index].output, self.model.output],
        )
        # 一時ファイルを削除
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def gradcam(self, x_train, x_i, x_name):
        plt.ioff()  # インタラクティブモードをオフ
        for c in range(self.class_length):
            # calc L_{Grad-CAM}^{class "c"}
            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = self.grad_model(
                    x_train.reshape(self.input_shape)
                )
                class_channel = preds[:, c]
            grads = tape.gradient(class_channel, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            cam = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
            cam = tf.maximum(tf.squeeze(cam), 0)
            cam = cam.numpy()

            self.cams[c] = np.vstack([self.cams[c], cam.reshape([-1])])

            # cam to heatmap
            heatmap = Image.fromarray(
                cam.reshape(
                    [self.last_conv_layer_shape[1], self.last_conv_layer_shape[2]]
                )
            )
            heatmap = heatmap.resize((self.input_shape[2], self.input_shape[1]))
            heatmap = np.array(heatmap)

            # plot
            fig = plt.figure()
            plt.imshow(np.rot90(heatmap), cmap="jet")
            plt.title(r"$L_{Grad-CAM}^{" + str(c + 1) + "}$ of " + x_name)
            plt.xticks(self.xticks_label, self.xticks_value)
            plt.yticks(self.yticks_label, self.yticks_value)
            plt.xlabel("frame")
            plt.ylabel("coefficient ranks")
            plt.colorbar()
            fig.savefig(
                "../target/img/" + str(x_i) + "(" + x_name + ")-" + str(c) + ".png",
                dpi=100,
            )
            plt.close(fig)

    def gradcam_mean(self):
        # calc mean of cam
        plt.ioff()  # インタラクティブモードをオフ
        for c in range(self.class_length):
            cams_mean = np.mean(self.cams[c], axis=0)
            heatmap = Image.fromarray(
                cams_mean.reshape(
                    [self.last_conv_layer_shape[1], self.last_conv_layer_shape[2]]
                )
            )
            heatmap = heatmap.resize((self.input_shape[2], self.input_shape[1]))
            heatmap = np.array(heatmap)
            fig = plt.figure()
            plt.imshow(np.rot90(heatmap), cmap="jet")
            plt.title("mean of " + r"$L_{Grad-CAM}^{" + str(c + 1) + "}$")
            plt.xticks(self.xticks_label, self.xticks_value)
            plt.yticks(self.yticks_label, self.yticks_value)
            plt.xlabel("frame")
            plt.ylabel("coefficient ranks")
            plt.colorbar()
            fig.savefig("../target/img/_mean-" + str(c) + ".png", dpi=100)
            plt.close(fig)
