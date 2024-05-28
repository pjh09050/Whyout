import pandas as pd
import numpy as np
from load_data import *
import warnings
warnings.filterwarnings("ignore")

class SGD():
    def __init__(self, R, k, learning_rate, reg_param, epochs, verbose=False):
        """
        :param R: rating matrix
        :param k: latent parameter
        :param learning_rate: alpha on weight update
        :param reg_param: beta on weight update
        :param epochs: training epochs
        :param verbose: print status
        """
        self._R = R
        self._num_users, self._num_items = R.shape
        self._k = k
        self._learning_rate = learning_rate
        self._reg_param = reg_param
        self._epochs = epochs
        self._verbose = verbose
        self.cost_list = []


    def fit(self):
        """
        training Matrix Factorization : Update matrix latent weight and bias

        참고: self._b에 대한 설명
        - global bias: input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용
        - 정규화 기능. 최종 rating에 음수가 들어가는 것 대신 latent feature에 음수가 포함되도록 해줌.

        :return: training_process
        """
        # init latent features
        self._U = np.random.normal(size=(self._num_users, self._k))
        self._V = np.random.normal(size=(self._num_items, self._k))

        # init biases
        self._b_U = np.zeros(self._num_users)
        self._b_V = np.zeros(self._num_items)
        self._b = np.mean(self._R[np.where(self._R != 0)])

        # train while epochs
        self._training_process = []
        for epoch in range(self._epochs):
            # rating이 존재하는 index를 기준으로 training
            xi, yi = self._R.nonzero()
            for i, j in zip(xi, yi):
                self.gradient_descent(i, j, self._R[i, j])
            cost = self.cost()
            self._training_process.append((epoch, cost))

            # print status
            if self._verbose == True and ((epoch + 1) % 10 == 0):
                self.cost_list.append(cost)
                print("Iteration: %d ; cost = %.4f" % (epoch + 1, cost))
        return self.cost_list


    def cost(self):
        """
        compute root mean square error
        :return: rmse cost
        """
        # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
        xi, yi = self._R.nonzero()
        # predicted = self.get_complete_matrix()
        cost = 0
        #print(len(xi), len(yi))
        count = 0
        for x, y in zip(xi, yi):
            count += 1
            cost += pow(self._R[x, y] - self.get_prediction(x, y), 2)
            # if self._R[x,y]== 6:
            #     print(cost, self._R[x,y], self.get_prediction(x,y))
        return np.sqrt(cost/len(xi))


    def gradient(self, error, i, j):
        """
        gradient of latent feature for GD
        :param error: rating - prediction error
        :param i: user index
        :param j: item index
        :return: gradient of latent feature tuple
        """
        du = (error * self._V[j, :]) - (self._reg_param * self._U[i, :]) # user에 대해 gradient -> item에 대해 미분
        dv = (error * self._U[i, :]) - (self._reg_param * self._V[j, :])
        return du, dv


    def gradient_descent(self, i, j, rating):
        """
        graident descent function
        :param i: user index of matrix
        :param j: item index of matrix
        :param rating: rating of (i,j)
        """
        # get error
        prediction = self.get_prediction(i, j)
        error = rating - prediction

        # update biases
        self._b_U[i] += self._learning_rate * (error - self._reg_param * self._b_U[i])
        self._b_V[j] += self._learning_rate * (error - self._reg_param * self._b_V[j])

        # update latent feature
        du, dv = self.gradient(error, i, j)
        self._U[i, :] += self._learning_rate * du
        self._V[j, :] += self._learning_rate * dv


    def get_prediction(self, i, j):
        """
        get predicted rating: user_i, item_j
        :return: prediction of r_ij
        """
        return self._b + self._b_U[i] + self._b_V[j] + self._U[i, :].dot(self._V[j, :].T)


    def get_complete_matrix(self):
        """
        computer complete matrix UXV + U.bias + V.bias + global bias

        - UXV 행렬에 b_U[:, np.newaxis]를 더하는 것은 각 열마다 bias를 더해주는 것
        - b_V[np.newaxis:, ]를 더하는 것은 각 행마다 bias를 더해주는 것
        - b를 더하는 것은 각 element마다 bias를 더해주는 것

        - newaxis: 차원을 추가해줌. 1차원인 Latent들로 2차원의 R에 행/열 단위 연산을 해주기위해 차원을 추가하는 것.

        :return: complete matrix R^
        """
        return self._b + self._b_U[:, np.newaxis] + self._b_V[np.newaxis:, ] + self._U.dot(self._V.T)
    
    def get_user_latent(self):
        return self._U
    
    def get_item_latent(self):
        return self._V
    
    def print_results(self):
        print("User Latent U:")
        print(self._U)
        print("Item Latent V:")
        print(self._V.T)
        print("U x V:")
        print(self._U.dot(self._V.T))
        print("bias:")
        print(self._b)
        print("User Latent bias:")
        print(self._b_U)
        print("Item Latent bias:")
        print(self._b_V)
        print("Final R matrix:")
        print(self.get_complete_matrix())
        print("Final RMSE:")
        print(self._training_process[self._epochs-1][1])

if __name__ == "__main__":
    # rating matrix - User X Item : (사용자 수 X 아이템 수)
    # U, V is (사용자 수 X k), (k X 아이템 수) matrix
    R = np.array(case2_user_place)
    factorizer = SGD(R, k=30, learning_rate=0.01, reg_param=0.01, epochs=2000, verbose=True)
    cost_list = factorizer.fit()
    complete_matrix = factorizer.get_complete_matrix()
    user_latent = factorizer.get_user_latent()
    item_latent = factorizer.get_item_latent()

    R1 = np.array(case2_user_product)
    factorizer = SGD(R1, k=30, learning_rate=0.01, reg_param=0.01, epochs=2000, verbose=True)
    cost_list1 = factorizer.fit()
    complete_matrix1 = factorizer.get_complete_matrix()
    user_latent1 = factorizer.get_user_latent()
    item_latent1 = factorizer.get_item_latent()

    R2 = np.array(case2_user_video)
    factorizer = SGD(R2, k=30, learning_rate=0.01, reg_param=0.01, epochs=2000, verbose=True)
    cost_list2 = factorizer.fit()
    complete_matrix2 = factorizer.get_complete_matrix()
    user_latent2 = factorizer.get_user_latent()
    item_latent2 = factorizer.get_item_latent()

    # 결과값 csv 파일로 저장
    df = pd.DataFrame(complete_matrix).astype(dtype='float16')
    df.to_csv('whyout_data/case2_sgd_rating_place.csv', index=False)
    df1 = pd.DataFrame(complete_matrix1).astype(dtype='float16')
    df1.to_csv('whyout_data/case2_sgd_rating_product.csv', index=False)
    df2 = pd.DataFrame(complete_matrix2).astype(dtype='float16')
    df2.to_csv('whyout_data/case2_sgd_rating_video.csv', index=False)

    u_latent = pd.DataFrame(user_latent).astype(dtype='float16')
    u_latent.to_csv('whyout_data/case2_user_latent_place.csv', index=False)
    u_latent1 = pd.DataFrame(user_latent1).astype(dtype='float16')
    u_latent1.to_csv('whyout_data/case2_user_latent_product.csv', index=False)
    u_latent2 = pd.DataFrame(user_latent2).astype(dtype='float16')
    u_latent2.to_csv('whyout_data/case2_user_latent_video.csv', index=False)
    
    i_latent = pd.DataFrame(item_latent).astype(dtype='float16')
    i_latent.to_csv('whyout_data/case2_item_latent_place.csv', index=False)
    i_latent1 = pd.DataFrame(item_latent1).astype(dtype='float16')
    i_latent1.to_csv('whyout_data/case2_item_latent_product.csv', index=False)
    i_latent2 = pd.DataFrame(item_latent2).astype(dtype='float16')
    i_latent2.to_csv('whyout_data/case2_item_latent_video.csv', index=False)