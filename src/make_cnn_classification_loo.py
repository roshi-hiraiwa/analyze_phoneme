# coding: utf_8
import numpy as np
import pandas as pd
from keras.utils import np_utils
import A_classification


# データの読み込み
csvdata = pd.read_csv("../target/train_data.csv", header=None)
x_train, y_train = np.split(csvdata, [240], axis=1)
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = x_train.reshape([-1, 40, 6, 1])
y_train = np_utils.to_categorical(y_train)  # class-index to one-hot-vector

csvdata = pd.read_csv("../train_data.csv", header=None, encoding="shift_jis")
string_train = csvdata.values[:, 0]
predicts = np.empty([0, y_train.shape[1]])

a_clf = A_classification.A_classification(y_train.shape[1])
a_clf.set_model()
a_clf.set_cams()


# leave-one-out(loo)
for i in range(len(x_train)):
    print(f"処理中 ({i+1}/{len(x_train)}): {string_train[i]}")

    a_clf.set_model()
    a_clf.fit_model(np.delete(x_train, i, 0), np.delete(y_train, i, 0))
    predicts = np.vstack([predicts, a_clf.prediction(x_train[i]).reshape([-1])])
    a_clf.set_grad_model()
    a_clf.gradcam(x_train[i], i, string_train[i])

# 最後に1回だけ保存
a_clf.save(
    "../target/summary.txt",
    "../target/summary.png",
    "../target/model.h5",
    "../target/loss.png",
)

a_clf.gradcam_mean()
result_array = np.concatenate(
    [string_train.reshape([-1, 1]), predicts.reshape([y_train.shape[0], -1])], axis=1
)
np.savetxt("../target/result.csv", result_array, delimiter=",", fmt="%s")
print("make_cnn_classification_loo.pyによってresult.csvが出力されました")

# 一時ファイルの削除
import os

if os.path.exists("t.h5"):
    os.remove("t.h5")
