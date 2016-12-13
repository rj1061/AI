import pandas
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model

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
        x = self.normalize(x)     #verify to see if this is correct
        y = self.normalizeY(y)

        # transpose the price matrix
        y = np.array([y]).T

        print x

        l = len(x)
        n = len(x[0])
        temp = np.ones(shape=(l, n))
        temp[:, 1] = x[:, 0]            # i think this might be the problem
        x = temp



        #initialise weights
        w = np.array([np.ones(n)]).T

        n = len(y)     # gradient descent

        for i in range(0, 10):
            J = np.sum((x.dot(w) - y) ** 2)/(2*n)

            print("Iteration %d, J(w): %f\n" % (i, J))

            gradient = np.dot(x.T, x.dot(w)-y)/n
            w = w - alpha * gradient

        return w

    def learn(self):
        train_values, train_labels = self.get_data()

        # -------------------------- Ridge Regression -------------------------
        print "\n Ridge Regression \n"
      #  reg = linear_model.Ridge(alpha = 0.25)
       # reg.fit(train_values, train_labels)

        # -------------------------- Linear Regression -------------------------
        print "\n Linear Regression \n"
        #reg = linear_model.LinearRegression()
        #reg.fit(train_values, train_labels)

        print "\n Linear Regression Implementation\n"
        w = self.LinearRegression(train_values, train_labels, .05)
        #print w

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
        train_csv = pandas.read_csv("train.csv", sep=',', header=None, nrows = 20)
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
            if x > 20:
                break
            x+=1

        for i in range(0,80):
            for j in range(1, train_csv.shape[0]):
                train_csv[i][j] = checkNaN(train_csv[i][j])

        return train_csv

Provider().learn()
