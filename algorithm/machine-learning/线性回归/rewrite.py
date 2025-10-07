from manim import *
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from typing import Optional

# 说明
"""
    @auther 巷北
    @time 2025.10.7 13:46
    @version 1.2

    昨天第一次写了一下,感觉还不错,但是debug方面做得不是很好,这次
    再优化一下, 添加更多接口, 以方便查看数据结构及形式.
"""
class MyLinearRegression:

    def __init__(
            self,
            file_name = None,
            ratio = 0.8
    ):
        if file_name is None:
            raise ValueTracker("请输入文件名.")
        np.set_printoptions(linewidth=np.inf)

        self.lines, self.header = self.init_data(file_name)
        self.split_data(ratio)
        self.standardscaler()
        self.split_x_y()

    def init_data(self, file_name):
        lines = np.loadtxt(file_name, delimiter = ',', dtype='str')
        header = lines[0]
        lines = lines[1:].astype(float)

        return lines, header

    def print_data_info(self):
        # 优化了一下输出结构,使意义,结构更加清晰明确.
        print(
            '数据特征: \n',
            *[
                f"x_{i}: {self.header[i]}, \n"
                for i in range(len(self.header) - 1)
            ]
            ,end = ''
        )
        print('数据标签: \n', f"y: {self.header[-1]}")
        print('数据总条数: ', len(self.lines))
        print('\n')

    def print_init_data_sample(self, num = 3):
        # 目的是输出几个初始化样本示例, 明确结构
        # 由于是仅查看结构,所以不打乱,从前往后输出
        lines = self.lines[:num]
        print("x_i infos: \n", lines[:, : -1])
        # print("\ny infos: \n", lines[:, -1]) # 本是二维数据,这门访问变成一维的了,不一致
        print("y infos: \n", lines[:, len(lines[0]) - 1:])
        print('\n')

    def split_data(self, ratio):
        split = int(len(self.lines)* ratio)
        np.random.seed(0)
        lines = np.random.permutation(self.lines)

        self.train= lines[:split]
        self.test = lines[split:]
    
    def print_split_data_length(self):

        print("train: ", len(self.train))
        print("test: ", len(self.test))
        print('\n')

    def test_permutation(self, testing_data: Optional[int|list] = None, num = None):
        # 第一个参数是测试数据, 自己任意输入
        # 第二个参数是测试当前self.lines的数据, num测试数量
        np.random.seed(0)
        if num is None:
            test_data = np.random.permutation(testing_data)
            print("initial data: \n", testing_data)

        else:
            lines = self.lines[:num]
            test_data = np.random.permutation(lines)
            print("initial data: \n", lines)

        print("permutation data:\n ", test_data)
        print("permutation data type: ", type(test_data))
        print('\n')

    def standardscaler(self):
        # 注意,fit了两次, 会以最后一次为准
        # 我这么做得目的是获取全部信息,方便输出对比分析.
        scaler = StandardScaler()
        scaler.fit(self.test)
        self.test_mean = scaler.mean_
        self.test_scale = scaler.scale_

        scaler.fit(self.train)
        self.train_mean = scaler.mean_
        self.train_scale = scaler.scale_


        self.train = scaler.transform(self.train)
        self.test = scaler.transform(self.test)

    def print_mean_and_scale(self):
        # 注意,此时自变量与因变量仍在一块
        print('train mean: ', self.train_mean)
        print("train scale: ", self.train_scale)
        print("test mean", self.test_mean)
        print("test scale: ", self.test_scale)
        print('\n')  

    def print_standard_data_sample(self, num = 3):
        print('train: \n', self.train[:num])
        print('test: \n', self.test[:num])
        print('\n')

    def split_x_y(self):
        self.x_train, self.y_train = self.train[:, :-1], self.train[:, -1].flatten()
        self.x_test, self.y_test = self.test[:, :-1], self.test[:, -1].flatten()    

    def print_split_x_y(self, num = 3):
        print("x_train: \n", self.x_train[:num])
        print("y_train: \n", self.y_train[:num])
        print("x_test: \n", self.x_test[:num])
        print('y_test: \n', self.y_test[:num])
        print('\n')
    
    def test_flatten(self, num = 3):
        # 注意到, train用到了flatten(),所以需要测试一下
        y_train = self.train[:num,-1]
        print('no flatten, only slicing: ', y_train)
        print("shape: ", y_train.shape)
        y_train = self.train[:num, -1].flatten()
        print("with flatten: ", y_train)
        print("shape: ", y_train.shape)

        print(
            """\n
                此时来看,毫无区别.但其实就是没区别.\n
                所以是否加flatten没有影响.那么关键点\n
                在哪呢?\n
                其实还是最基本的.通过[:, -1],本身就是\n
                获取最后一列, 并且自动变为1为数组.\n
                shape为何是(x, )?后面为空代表没有列,\n
                也就是一维数组.对于二维数组,shape才\n
                有(x,y)\n
                下面我们再测试y为二维的情况\n
                \n
            """
        )

        y_train = self.train[:num, len(self.train[0])-1:]
        print('no flatten, only slicing: \n', y_train)
        print("shape: ", y_train.shape)

        y_train = self.train[:num, len(self.train[0])-1:].flatten()
        print("with flatten: ", y_train)
        print("shape: ", y_train.shape)
        print('\n')

        # 好了,又多了一个知识点.单独取出来某列,会自动变为flatten.
    @staticmethod
    def get_X(data):
        t = np.ones((len(data), 1))
        X = np.hstack([data, t])
        return X

    def test_ones(self, param = 4, num = 3):
        # 确实得单独研究.
        # 传入一个整除,返回全1数组
        # 传入一个shape,返回对应
        # 形状的全1数组.
        t = np.ones(param)
        print(f"parameter {str(param)}: \n", t)
        print('--------------------------')
        # 上面的其实非常有意思.接下来具体测试param了,收获颇丰
        x_train = self.x_train[:num]
        print("行数1 len(x_train): ", len(x_train))
        print("行数2 x_train.shape[0]: ", x_train.shape[0])
        print("列数1 len(x_train[0]): ", len(x_train[0]))
        print('列数2 x_train.shape[1]: ', x_train.shape[1])
        print(
            """\n
                十分清晰了, 源代码中使用np.ones((len(x_train), 1)),\n
                m目的就是获取n行1列的全1数组.返回的是二维形式.当然,我们\n
                也不止只有一种访问方式.通过.shape[0]也可以访问.
                \n
            """
        )
        print('\n')

    def test_concatenate(self, data):
        print(
            """
                虽说是测试concatenate的,但是我跟喜欢用vstack和hstack,所以\n
                就用这两个代替了.
            """
        )

        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        t = np.ones((data.shape[0], 1))
        X = np.hstack([data, t])

        print('拼接列: \n', X)
        print('shape: ',X.shape)

        t = np.ones((1, data.shape[1]))
        X = np.vstack([data, t])
        
        print('拼接行: \n', X)
        print("shape: ", X.shape)
        print("至于拼接行或拼接列的几何意义,自己去分析楼~")
        print('\n')
    
    def get_theta(self, X, y):
        return np.linalg.inv(X.T @ X) @ X.T @ y
    
    def print_theta(self):
        X = self.get_X(self.x_train)
        theta = self.get_theta(X, self.y_train)
        print("x_i theta 回归系数: ", theta[: -1])
        print("常数项 theta 回归系数:", theta[len(theta) - 1 :])
        print('\n')

    def get_RMSE(self, y_test, y_pred):
        return np.sqrt(np.square(y_test - y_pred).mean())

    def print_RMSE(self):
        # 先求系数, 然后再进行预测, 最后看看效果如何
        X = self.get_X(self.x_train)
        theta = self.get_theta(X, self.y_train)

        X_test = self.get_X(self.x_test)
        y_pred = X_test @ theta

        rmse_loss = self.get_RMSE(self.y_test, y_pred)
        print("RMSE: ", rmse_loss)
        print('\n')

    def print_LinearRegression_RMSE(self):
        linreg = LinearRegression()
        linreg.fit(self.x_train, self.y_train)

        y_pred = linreg.predict(self.x_test)
        rmse_loss = self.get_RMSE(self.y_test, y_pred)

        print("x_i theta 回归系数: ",linreg.coef_,)
        print("常数项 theta 回归系数: ", linreg.intercept_)
        print("RMSE: ", rmse_loss)
        print('\n')

    # 好了,上面的基本步骤就完成了.不过有些函数名字太复杂了,而且下面都是些测试
    # 输出,里面有很多的print,眼花缭乱地.所以我重新命名一下接口,使得用户能直接
    # 查看数据.

    def get_train(self, num):
        return self.train[:num]
    def get_test(self, num):
        return self.test[:num]
    def get_x_train(self, num):
        return self.x_train[:num]
    def get_x_test(self, num):
        return self.x_test[:num]
    def get_y_train(self, num):
        return self.y_train[:num]
    def get_y_test(self, num):
        return self.y_test[:num]
    def get_theta_and_RMSE(self):
        X = self.get_X(self.x_train)
        theta = self.get_theta(X, self.y_train)

        X_test = self.get_X(self.x_test)
        y_pred = X_test @ theta

        rmse_loss = self.get_RMSE(self.y_test, y_pred)
        return theta, rmse_loss
    def get_LinearRegression_theta_and_RMSE(self):
        linreg = LinearRegression()
        linreg.fit(self.x_train, self.y_train)

        y_pred = linreg.predict(self.x_test)
        rmse_loss = self.get_RMSE(self.y_test, y_pred)
        theta = np.append(linreg.coef_, linreg.intercept_)

        return theta, rmse_loss
    
    # 好了,上面的基础操作弄得差不多了,明天再仔细弄弄算法部分.
    

m = MyLinearRegression('A.csv')
# m.print_data_info()
# m.print_init_data_sample(3)
# m.print_split_data_length()
# m.test_permutation(num = 7)
# m.print_standard_data_sample(3)
# m.print_mean_and_scale()
# m.print_split_x_y(3)
# m.test_flatten()
# m.test_ones()
# m.test_concatenate([
#     [1, 2, 3 ,4, 5],
#     [1, 2, 3, 4 ,5],
#     [5, 3, 2, 4, 1],
#     [4, 2, 1, 3, 7]
# ])
# m.print_theta()
# m.print_RMSE()
# m.print_LinearRegression_RMSE()

print(m.get_LinearRegression_theta_and_RMSE()[0])
print(m.get_LinearRegression_theta_and_RMSE()[1])
print(m.get_theta_and_RMSE()[0])
print(m.get_theta_and_RMSE()[1])