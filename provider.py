import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model

class Provider:

    def learn(self):
        train_values, train_labels = self.get_data()

        # -------------------------- Ridge Regression -------------------------
        print "\n Ridge Regression \n"
        reg = linear_model.Ridge(alpha = 0.25)
        reg.fit(train_values, train_labels)

        # -------------------------- Linear Regression -------------------------
        print "\n Linear Regression \n"
        reg = linear_model.LinearRegression()
        reg.fit(train_values, train_labels)

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

        return train_csv

Provider().learn()
