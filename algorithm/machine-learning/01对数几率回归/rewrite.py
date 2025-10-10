import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LogisticRegression

class MyLogistic:
    def __init__(
            self,
            file_name = None,
            ratio = 0.7,
            suppress = True
    ):
        if file_name is None:
            raise TypeError("请输入文件地址.")

        np.set_printoptions(linewidth=np.inf, suppress=suppress)
        
        self.x_total, self.y_total = self.load_file(file_name)
        self.x_train, self.y_train, self.x_test, self.y_test = self.split_x_y(ratio)



    def load_file(self, file_name):
        lines = np.loadtxt(file_name, delimiter = ',', dtype = float)
        # x_total = lines[:, :2]]
        # y_total = lines[:, len(lines[0]) - 1:]# 之前多了个flatten,但你这么写,必须要加
        # y_total = lines[:, len(lines[0]) - 1:].flatten()

        # 对于len(lines[0]) 获取的是矩阵列数. 矩阵列数还可以通过shape[1]获取.没必要过于死板.
        # ]

        x_total = lines[:, : -1]
        y_total = lines[:, lines.shape[1] - 1:].flatten()

        return x_total, y_total

    def print_data_infos(self, num = 3):
        print("查看x_i相关信息: \n", self.x_total[:num])
        print("查看y相关信息: \n", self.y_total[:num])

    def get_init_plot(self):
        pos_index = np.where(self.y_total == 1)
        neg_index = np.where(self.y_total == 0)

        plt.scatter(
            self.x_total[pos_index, 0],
            self.x_total[pos_index, 1],
            marker = 'o',
            color = 'coral',
            s = 10
        )
        
        plt.scatter(
            self.x_total[neg_index, 0],
            self.x_total[neg_index, 1],
            marker = 'x',
            color = 'blue',
            s = 10
        )

        plt.xlabel("X1 axis")
        plt.ylabel('Xx axis')
        plt.show()

    def split_x_y(self, ratio):
        np.random.seed(0)
        split = int(len(self.x_total) * ratio)
        idx = np.random.permutation(len(self.x_total))

        self.x_total = self.x_total[idx]
        self.y_total = self.y_total[idx]

        x_train, y_train = self.x_total[:split], self.y_total[:split]
        x_test, y_test = self.x_total[split: ], self.y_total[split :]

        return x_train, y_train, x_test, y_test
    
    def print_split_x_y(self, num = 3):
        print("查看打乱后的x_total: \n", self.x_total[:num])
        print("查看打乱后的y_total: \n", self.y_total[:num])

    @staticmethod
    def acc(y_true, y_pred):
        return np.mean(y_true == y_pred) # 很聪明的写法. 但是Python列表并不支持这样做, 只会判断列表是否相等, 而不是列表内部元素. 本质是重载了==
    
    @staticmethod
    def auc(y_true, y_pred):

        idx = np.argsort(y_pred)[::-1]
        y_true = y_true[idx]
        y_pred = y_pred[idx]

        tp = np.cumsum(y_true)
        fp = np.cumsum(1 - y_true)
        tpr = tp / tp[-1]
        fpr = fp / fp[-1]

        s = 0.0
        tpr = np.hstack([[0], tpr])
        fpr = np.hstack([[0], fpr])

        for i in range(1, len(fpr)):
            s += (fpr[i] - fpr[i - 1]) * tpr[i]

        return s

    @staticmethod
    def logistic(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def get_X(data):
        t = np.ones((data.shape[0], 1))
        X = np.hstack([data, t])
        return X
    
    def GD(
            self,
            num_steps,
            learning_rate, 
            l2_coef,
    ):
        X = self.get_X(self.x_train)
        X_test = self.get_X(self.x_test)
        theta = np.random.normal(size = (X.shape[1], ))

        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []
        train_auc = []
        test_auc = []

        for _ in range(num_steps):

            pred = self.logistic(X @ theta)
            grad = -X.T @ (self.y_train - pred) + l2_coef * theta

            theta -= learning_rate * grad

            train_loss = - self.y_train.T @ np.log(pred) \
                        - (1 - self.y_train).T @ np.log(1 - pred) \
                        + l2_coef * np.linalg.norm(theta) ** 2 / 2
            train_losses.append(train_loss / len(X))

            test_pred = self.logistic(X_test @ theta)
            test_loss = - self.y_test.T @ np.log(test_pred) \
                        - (1 - self.y_test).T @ np.log(1 - test_pred)
            test_losses.append(test_loss/ len(X_test))

            train_acc.append(self.acc(self.y_train, pred >= 0.5))
            test_acc.append(self.acc(self.y_test, test_pred >= 0.5))
            train_auc.append(self.auc(self.y_train, pred))
            test_auc.append(self.auc(self.y_test, test_pred))

        return theta, train_losses, test_losses, \
                train_acc, test_acc, train_auc, test_auc, \
                X,X_test
    
    def invoke_GD(self, num_steps, learning_rate, l2_coef):

        self.num_steps = num_steps
        self.theta, self.train_losses, self.test_losses, \
        self.train_acc, self.test_acc, self.train_auc, self.test_auc, \
        self.X, self.X_test = self.GD(num_steps, learning_rate, l2_coef)

    def print_test_accuracy(self):
        y_pred = np.where(self.logistic(self.X_test @ self.theta) >= 0.5, 1, 0)
        final_acc = self.acc(self.y_test, y_pred)
        print('预测准确率: ',final_acc)
        print("回归系数: ", self.theta)

    def _first_plot(self, xticks):
        plt.subplot(221)
        plt.plot(xticks, self.train_losses, color = 'blue', label = 'train loss')
        plt.plot(xticks, self.test_losses, color = 'red', ls = '--', label = 'test loss')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
    def _second_plot(self, xticks):
        plt.subplot(222)
        plt.plot(xticks, self.train_acc, color = 'blue', label = 'train accuracy')
        plt.plot(xticks, self.test_acc, color = 'red', ls = '--', label = 'test accrucy')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

    def _third_plot(self, xticks):
        plt.subplot(223)
        plt.plot(xticks, self.train_auc, color = 'blue', label = 'train AUC')
        plt.plot(xticks, self.test_auc, color = 'red', ls = '--', label = 'test AUC')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()

    def _fourth_plot(self):
        plt.subplot(224)
        plot_x = np.linspace(-1.1, 1.1, 100)
        plot_y = -(self.theta[0] * plot_x + self.theta[2]) / self.theta[1]
        pos_index = np.where(self.y_total == 1)
        neg_index = np.where(self.y_total == 0)

        plt.scatter(
            self.x_total[pos_index, 0],
            self.x_total[pos_index, 1],
            marker='o',
            color = 'coral',
            s = 10,
        )
        plt.scatter(
            self.x_total[neg_index, 0], 
            self.x_total[neg_index, 1],
            marker = 'x',
            color = 'blue',
            s = 10
        )

        plt.plot(plot_x, plot_y, ls='-.', color = 'green')
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.xlabel('X1 axis')
        plt.ylabel('X2 axis')

    def set_position(self):

        mgr = plt.get_current_fig_manager()
        screen = mgr.window.screen()          
        screen_geom = screen.geometry()
        fig_width, fig_height = 1000, 750      
        x = (screen_geom.width() - fig_width) // 2
        y = (screen_geom.height() - fig_height) // 2
        mgr.window.setGeometry(x, y, fig_width, fig_height)

    def get_four_plot(self, issave = False):
        plt.figure(figsize=(13, 9))
        xticks = np.arange(self.num_steps) + 1

        self._first_plot(xticks)
        self._second_plot(xticks)
        self._third_plot(xticks)
        self._fourth_plot()
        
        if issave:
            plt.savefig('output.png')
            plt.savefig('output.pdf')

        self.set_position()

        plt.show()

    def print_LogisticRegression_accuracy(self):
        lr_clf = LogisticRegression(solver = 'liblinear')
        lr_clf.fit(self.x_train, self.y_train)
        print("回归系数: ", lr_clf.coef_[0], lr_clf.intercept_)

        y_pred = lr_clf.predict(self.x_test)
        print('准确率为: ', np.mean(y_pred == self.y_test))



m = MyLogistic('Lr_dataset.csv')
num_steps = 250
learning_rate = 0.002
l2_coef = 1.0
np.random.seed(0)
m.print_data_infos(3)
# m.get_init_plot()
# m.print_split_x_y(10) 

# m.invoke_GD(num_steps, learning_rate, l2_coef)
# m.print_test_accuracy()
# m.get_four_plot()
# m.print_LogisticRegression_accuracy()
