# coding:utf-8
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import mpl
from pathlib import Path

mpl.rcParams['font.sans-serif'] = ['SimHei']

# 图片矢量化
def img2vector(image):
    img = cv2.imread(image, 0)  # 灰度读取
    if img is None:
        print(f"⚠️ 无法读取图片: {image}")
        return np.zeros((1, 112*92), dtype=np.float32)  # 占位，避免崩溃
    imgVector = img.reshape(1, -1).astype(np.float32)
    return imgVector

# 数据集路径
orlpath = str(Path(__file__).resolve().parent / "dataset" / "bmp")

# 读入人脸库
def load_orl(k):
    train_face = np.zeros((40 * k, 112 * 92), dtype=np.float32)
    train_label = np.zeros(40 * k)
    test_face = np.zeros((40 * (10 - k), 112 * 92), dtype=np.float32)
    test_label = np.zeros(40 * (10 - k))
    
    for i in range(40):  # 40个人
        people_num = i + 1
        sample = np.random.permutation(10) + 1  # 随机选择 1~10 的图片
        for j in range(10):
            image = os.path.join(orlpath, f"s{people_num}", f"{sample[j]}.BMP")
            img = img2vector(image)
            if j < k:
                train_face[i * k + j, :] = img
                train_label[i * k + j] = people_num
            else:
                test_face[i * (10 - k) + (j - k), :] = img
                test_label[i * (10 - k) + (j - k)] = people_num

    return train_face, train_label, test_face, test_label

# PCA 算法
def PCA(data, r):
    data = np.asarray(data, dtype=np.float32)
    rows, cols = data.shape
    data_mean = np.mean(data, axis=0)
    A = data - np.tile(data_mean, (rows, 1))
    
    # 协方差矩阵
    C = A @ A.T
    D, V = np.linalg.eig(C)
    
    # 前 r 个特征向量
    V_r = V[:, :r]
    V_r = A.T @ V_r
    for i in range(r):
        V_r[:, i] /= np.linalg.norm(V_r[:, i])
    
    final_data = A @ V_r
    return final_data, data_mean, V_r

# 人脸识别
def face_rec():
    for r in range(10, 41, 10):
        print(f"当降维到{r}维时")
        x_value = []
        y_value = []

        for k in range(1, 10):
            train_face, train_label, test_face, test_label = load_orl(k)
            data_train_new, data_mean, V_r = PCA(train_face, r)

            num_train = data_train_new.shape[0]
            num_test = test_face.shape[0]

            # 投影测试集
            temp_face = test_face - np.tile(data_mean, (num_test, 1))
            data_test_new = temp_face @ V_r

            # 测试准确度
            true_num = 0
            for i in range(num_test):
                testFace = data_test_new[i, :]
                diffMat = data_train_new - np.tile(testFace, (num_train, 1))
                sqDistances = np.sum(diffMat**2, axis=1)
                indexMin = np.argmin(sqDistances)
                if train_label[indexMin] == test_label[i]:
                    true_num += 1

            accuracy = true_num / num_test
            x_value.append(k)
            y_value.append(round(accuracy, 2))
            print(f"当每个人选择{k}张照片进行训练时，The classify accuracy is: {accuracy*100:.2f}%")

        # 绘制每个 r 的准确率
        plt.plot(x_value, y_value, marker="o", markerfacecolor="red")
        for a, b in zip(x_value, y_value):
            plt.text(a, b, f"{b}", ha='center', va='bottom', fontsize=10)
        plt.title(f"降到{r}维时识别准确率", fontsize=14)
        plt.xlabel("K值", fontsize=14)
        plt.ylabel("准确率", fontsize=14)
        plt.show()

if __name__ == '__main__':
    face_rec()
