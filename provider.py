import pandas
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
from sklearn import preprocessing

def findMax(x):
    max = x[0]
    for i in range(1, len(x)-1):
        if max < x[i]:
            max = x[i]
    return max


def findMin(x):
    min = x[0]
    for i in range(1, len(x) - 1):
        if min > x[i]:
            min = x[i]
    return min

def checkNaN(value):
    if (pandas.isnull(value)):
        return 0
    return value

class Provider:
        # revisit normalization methodology
    def normalize(self, x):
        for i in range(0, len(x)-1):
            max = findMax(x[i])
            min = findMin(x[i])
            for j in range(0, len(x[i])-1):
                x[i][j] = (x[i][j]-min)/(max-min)
        return x

    def normalizeY(self, y):
        max = findMax(y)
        min = findMin(y)
        for i in range(0, len(y)-1):
            y[i] = (y[i] - min) / (max - min)
        return y

    def LinearRegression(self, x, y, alpha):
        x = preprocessing.normalize(x)     # Verify to see if this is correct
        y = preprocessing.normalize(y)

        # transpose the price matrix
        y = y.T

        l = len(x)
        n = len(x[0])
        x = np.append(x, np.ones(shape=(l,1)),axis=1)



        # Initialize weights
        w = np.random.rand(1, n + 1).T

        n = len(y)     # Gradient Descent

        for i in range(0, 10):
            inner = x.dot(w)
            J = np.sum((inner - y) ** 2)/(2*n)

            print("Iteration %d, J(w): %f\n" % (i, J))

            df = x.T

            gradient = df.dot(x.dot(w)-y)/n #np.dot(df, x.dot(w)-y)/n
            w = w - alpha * gradient

        return w

    def RidgeRegression(self, x, y, alpha, penalty):
        x = preprocessing.normalize(x)  # Verify to see if this is correct
        y = preprocessing.normalize(y)

        # transpose the price matrix
        y = y.T

        l = len(x)
        n = len(x[0])
        x = np.append(x, np.ones(shape=(l, 1)), axis=1)

        # Initialize weights
        w = np.random.rand(1, n + 1).T

        n = len(y)  # Gradient Descent

        for i in range(0, 10):
            inner = x.dot(w)
            J = np.sum((inner - y) ** 2) / (2 * n)

            print("Iteration %d, J(w): %f\n" % (i, J))

            df = x.T

            gradient = df.dot(x.dot(w) - y) / n  # np.dot(df, x.dot(w)-y)/n
            lasso = w ** 2
            w = w - (alpha * gradient)-(penalty * lasso)

        return w

    def learn(self):
        train_values, train_labels = self.get_data()

        print "\n Linear Regression Implementation\n"
        self.LinearRegression(train_values, train_labels, .1)
        reg = linear_model.LinearRegression()
        print reg.fit(train_values, train_labels).score(train_values, train_labels)

        print "\n Ridge Regression Implementation\n"
        self.RidgeRegression(train_values, train_labels, .1, 0.1)
        reg = linear_model.Ridge(alpha=0.25)
        print reg.fit(train_values, train_labels).score(train_values, train_labels)

    def get_data(self):
        train_labels = []
        train_values =[]

        train_csv = self.get_data_csv()

        for index in range(1, train_csv.shape[0]):
            train_values.append({train_csv[key][0]: train_csv[key][index] for key in range(0, train_csv.shape[1] - 1)})
            train_labels.append(train_csv[train_csv.shape[1]-1][index])

        train_vector_maker = DictVectorizer()
        train_final = train_vector_maker.fit_transform(train_values).toarray()

        return train_final, train_labels

    def get_data_csv(self):
        # Read from csv
        train_csv = pandas.read_csv("train.csv", sep=',', header=None)
        x = 0
        for p in range(1, train_csv.shape[0]):
            train_csv[0][p] = float(train_csv[0][p])  # ID
            train_csv[1][p] = float(train_csv[1][p])  # MSSubClass
            train_csv[3][p] = float(train_csv[3][p])  # LotFrontage
            train_csv[4][p] = float(train_csv[4][p])  # LotArea
            train_csv[17][p] = float(train_csv[17][p])  # OverallQual
            train_csv[18][p] = float(train_csv[18][p])  # OverallCond
            train_csv[19][p] = float(train_csv[19][p])  # YearBuilt
            train_csv[20][p] = float(train_csv[20][p])  # YearRemodAdd
            train_csv[26][p] = float(train_csv[26][p])  # MasVnrArea
            train_csv[34][p] = float(train_csv[34][p])  # BsmtFinSF1
            train_csv[36][p] = float(train_csv[36][p])  # BsmtFinSF2
            train_csv[37][p] = float(train_csv[37][p])  # BsmtUnfSF
            train_csv[38][p] = float(train_csv[38][p])  # TotalBsmtSF
            train_csv[43][p] = float(train_csv[43][p])  # 1stFlrSF
            train_csv[44][p] = float(train_csv[44][p])  # 2stFlrSF
            train_csv[45][p] = float(train_csv[45][p])  # LowQualFinSF
            train_csv[46][p] = float(train_csv[46][p])  # GrLivArea
            train_csv[47][p] = float(train_csv[47][p])  # BsmtFullBath
            train_csv[48][p] = float(train_csv[48][p])  # BsmtHalfBath
            train_csv[49][p] = float(train_csv[49][p])  # FullBath
            train_csv[50][p] = float(train_csv[50][p])  # HalfBath
            train_csv[51][p] = float(train_csv[51][p])  # BedroomAbvGr
            train_csv[52][p] = float(train_csv[52][p])  # KitchenAbvGr
            train_csv[54][p] = float(train_csv[54][p])  # TotRmsAbvGrd
            train_csv[56][p] = float(train_csv[56][p])  # Fireplaces
            train_csv[59][p] = float(train_csv[59][p])  # GarageYrBlt
            train_csv[61][p] = float(train_csv[61][p])  # GarageCars
            train_csv[62][p] = float(train_csv[62][p])  # GarageArea
            train_csv[66][p] = float(train_csv[66][p])  # WoodDeckSF
            train_csv[67][p] = float(train_csv[67][p])  # OpenPorchSF
            train_csv[68][p] = float(train_csv[68][p])  # EnclosedPorch
            train_csv[69][p] = float(train_csv[69][p])  # 3SsnPorch
            train_csv[70][p] = float(train_csv[70][p])  # ScreenPorch
            train_csv[71][p] = float(train_csv[71][p])  # PoolArea
            train_csv[75][p] = float(train_csv[75][p])  # MiscVal
            train_csv[76][p] = float(train_csv[76][p])  # MoSold
            train_csv[77][p] = float(train_csv[77][p])  # YrSold
            train_csv[80][p] = float(train_csv[80][p])  # SalePrice

        for i in range(0,80):
            for j in range(1, train_csv.shape[0]):
                train_csv[i][j] = checkNaN(train_csv[i][j])

        return train_csv

Provider().learn()
