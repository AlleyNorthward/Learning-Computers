from manim import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

class MyLinearRegression:
    def __init__(
            self,
            file_name = None,
            show_init_info = False,
            ratio = 0.8
    ):
        if file_name is None:
            raise TypeError("请传入文件名.")
        
        self.init_datas(file_name, show_init_info, ratio)
    
    def init_datas(
            self, 
            file_name, 
            show_init_info,
            ratio,
    ):
        lines = np.loadtxt(file_name, delimiter=',', dtype='str')
        header = lines[0]
        lines = lines[1:].astype(float)
        
        if show_init_info:
            print('数据特征(多元自变量x_i):', ', '.join(header[: -1]))
            print('数据标签(价格y): ', header[-1])
            print('数据总条数: ', len(lines))

        split = int(len(lines) * ratio)
        np.random.seed(0)
        lines = np.random.permutation(lines)
        train, test = lines[:split], lines[split:]

        scaler = StandardScaler()
        scaler.fit(train)
        train = scaler.transform(train)
        test = scaler.transform(test)

        # 由于返回值太多,在init中初始化太长了,直接就在这里初始化了.
        self.x_train, self.y_train = train[:, :-1], train[:,-1].flatten()
        self.x_test, self.y_test = test[:, :-1], test[:, -1].flatten()

    def get_theta(self):
        X = self._init_X(self.x_train)
        theta = np.linalg.inv(X.T @ X) @ X.T @ self.y_train
        return theta
    
    def _calculator_rmse_loss(self,y_test, y_pred):
        return np.sqrt(np.square(y_test - y_pred).mean())
    
    def _init_X(self, x):
        t = np.ones((len(x), 1))
        return np.hstack([x, t])
        
    def get_RMSE(self):
        X_test = self._init_X(self.x_test)
        theta = self.get_theta()
        y_pred = X_test @ theta

        rmse_loss = self._calculator_rmse_loss(self.y_test, y_pred)
        return rmse_loss
    
    def get_LinearRegression_RMSE(self, isshow_theta = False):
        linreg = LinearRegression()
        linreg.fit(self.x_train, self.y_train)

        if isshow_theta:
            print("回归系数: ", linreg.coef_, linreg.intercept_)

        y_pred = linreg.predict(self.x_test)

        rmse_lose = self._calculator_rmse_loss(self.y_test, y_pred)
        return rmse_lose

    def _batch_generator(self, x, y, batch_size, shuffle = True):
        batch_count = 0
        if shuffle:
            idx = np.random.permutation(len(x))
            x = x[idx]
            y = y[idx]

        while True:
            start = batch_count * batch_size
            end = min(start + batch_size, len(x))

            if start >= end:
                break
            batch_count += 1
            yield x[start: end], y[start: end]

    def _SGD(self, num_epoch, learning_rate, batch_size):
        X = self._init_X(self.x_train)
        X_test = self._init_X(self.x_test)

        theta = np.random.normal(size = X.shape[1])

        train_losses = []
        test_losses  = []
        
        for _ in range(num_epoch):
            batch_g = self._batch_generator(X, self.y_train, batch_size)
            train_loss = 0
            for x_batch, y_batch in batch_g:
                grad = x_batch.T @ (x_batch @ theta - y_batch)

                theta = theta - learning_rate * grad / len(x_batch)

                train_loss += np.square(x_batch @ theta - y_batch).sum()

            train_loss = np.sqrt(train_loss / len(X))
            train_losses.append(train_loss)

            test_loss = self._calculator_rmse_loss(X_test @ theta, self.y_test)
            test_losses.append(test_loss)

            self.train_losses = train_losses
            self.test_losses = test_losses
    
        return theta
    
    def get_SGD_plot(self, num_epoch, learning_rate, batch_size):
        np.random.seed(0)
        self._SGD(num_epoch, learning_rate,batch_size)

        plt.plot(np.arange(num_epoch), self.train_losses, color = 'blue', label = 'train loss')
        plt.plot(np.arange(num_epoch), self.test_losses, color = 'red', ls = '--', label = 'test loss')

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer = True))
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.legend()
        plt.show()
        


test = MyLinearRegression('A.csv')
print(test.get_theta())
print(test.get_RMSE())
print(test.get_LinearRegression_RMSE(isshow_theta=True))
num_epoch, learning_rate, batch_size = 20, 0.01, 32
test.get_SGD_plot(num_epoch, learning_rate, batch_size)



class Test(Scene):
    def setup(self):
        self.camera.background_color = "#cee"
        self.axes = Axes(
            tips = False,
            axis_config={
                "include_numbers": False,
                "include_ticks": False
            }
        ).set_color(BLACK)
    def construct(self):
        self.add(self.axes)