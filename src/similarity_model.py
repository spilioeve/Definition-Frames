import numpy as np
import random
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from scipy import stats
import pdb
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

class Similarity:

    def __init__(self, embedFile):
        self.embeddings, self.dim =self.readEmbeddings(embedFile)
        #self.baseline, self.baseDim=self.readEmbeddings(baseline)
        #self.vocabulary={}


    def loadData(self, datasets):
        pairCount=0.
        pairs={}
        for d in datasets:
            f=open('../data/word-sim/'+d)
            lines=f.read().split('\n')[:-1]
            f.close()
            norm=0.0
            data_pairs={}
            for line in lines:
                word1, word2, score= line.split()
                score=float(score)
                if score>norm: norm= score
                data_pairs[(word1, word2)]= score
            pairs.update({i:data_pairs[i]/norm for i in data_pairs})
        X = np.zeros((2*self.dim))
        y = []
        pairWords=list(pairs.keys())
        random.shuffle(pairWords)
        for word1, word2 in pairWords:
            if word1 in self.embeddings and word2 in self.embeddings:
                v1 = np.array(self.embeddings[word1])
                v2 = np.array(self.embeddings[word2])
                #x_i= np.outer(v1, v2).flatten()
                x_1 = np.concatenate((v1, v2))
                x_2 = np.concatenate((v2, v1))
                X = np.vstack((X, x_1))
                X = np.vstack((X, x_2))
                y.append(pairs[(word1, word2)])
                y.append(pairs[(word1, word2)])
                pairCount+=1
        X = X[1:, :]
        Y = np.array(y)
        print('Number of pairs found: '+str(pairCount)+ '/' +str(len(pairs)))
        return X, Y

    def readEmbeddings(self, embedFile):
        f=open('../frame_embeddings/'+embedFile)
        lines=f.read().split('\n')[:-1]
        f.close()
        embeddings={}
        for item in lines:
            item= item.split(' ')
            term=item[0]
            vector=[float(i) for i in item[1:]]
            dim= len(vector)
            embeddings[term]=vector
        return embeddings, dim

    def run_Regressions(self, datasets):
        X, Y=self.loadData(datasets)
        num_data= X.shape[0]
        if num_data<10: return

        train_X= X[:int(num_data*0.8),:]
        train_y= Y[:int(num_data*0.8)]
        test_X= X[int(num_data*0.8):, :]
        test_y= Y[int(num_data*0.8):]
        linear = LinearRegression().fit(train_X, train_y)
        train_r2= linear.score(train_X, train_y)
        r2= linear.score(test_X, test_y)
        pred_y= linear.predict(test_X)
        error= mean_squared_error(test_y, pred_y)
        print('Linear Reg: ' + str(error.mean()) + '\t' + str(r2.mean()))
        weight = linear.coef_
        bias= linear.intercept_
        rho, pval = stats.spearmanr(pred_y, test_y)
        print('Rho, Pval: ' + str(rho) + '\t' + str(pval))

        # print('Trial Scores')
        # print(train_score, test_score)
        # error = cross_val_score(linear_regr, X, Y, cv=5, scoring='neg_mean_squared_error')
        # r2 = cross_val_score(linear_regr, X, Y, cv=5, scoring='r2')
        # print('Linear Reg: '+ str(-1*error.mean())+ '\t'+ str(r2.mean()))
        # ridge_regr = Ridge(alpha=0.6)
        # error = cross_val_score(ridge_regr, X, Y, cv=5, scoring='neg_mean_squared_error')
        # r2 = cross_val_score(ridge_regr, X, Y, cv=5, scoring='r2')
        # print('Ridge: '+ str(-1*error.mean())+ '\t'+ str(r2.mean()))
        # lasso = Lasso(alpha=0.6)
        # error = cross_val_score(lasso, X, Y, cv=5, scoring='neg_mean_squared_error')
        # r2 = cross_val_score(lasso, X, Y, cv=5, scoring='r2')
        # print('Lasso: '+ str(-1*error.mean())+'\t'+str(r2.mean()))


    def neuralSimilarity(self, datasets, epochs):
        X, Y = self.loadData(datasets)
        #X_, Y_ =self.loadBaseline(datasets)
        num_data = X.shape[0]
        kf = KFold(n_splits=10)
        test_rho=np.array(())
        test_p = np.array(())
        init_rho = np.array(())
        init_p = np.array(())
        if num_data < 100: return
        for train, test in kf.split(X):
            train_X= X[train]
            train_y= Y[train]
            test_X= X[test]
            test_y=Y[test]
            # train_X= X[:int(num_data*0.8),:]
            # train_y= Y[:int(num_data*0.8)]
            # test_X= X[int(num_data*0.8):, :]
            # test_y= Y[int(num_data*0.8):]
            #model = nn.Sequential(nn.Linear(self.dim, int(self.dim/4)), nn.ReLU(), nn.Linear(int(self.dim/4), self.dim))
            model= nn.Linear(self.dim, self.dim)
            cos = nn.CosineSimilarity()
            loss = nn.MSELoss()
            #loss = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.03, weight_decay=1e-4)
            for epoch in range(epochs):
                # print("epoch: " + str(epoch))
                y_pred = np.array(())
                ##RAndomize shuffle X, Y
                randomize = np.arange(len(train_y))
                np.random.shuffle(randomize)
                train_X= train_X[randomize]
                train_y= train_y[randomize]
                for i in range(len(train_X)):
                    model.zero_grad()
                    x_1= train_X[i][:self.dim]
                    x_2 = train_X[i][self.dim:]
                    y= train_y[i]
                    pred_x1= model(torch.tensor(x_1, dtype=torch.float32)).view((1, self.dim))
                    pred_x2 = model(torch.tensor(x_2, dtype=torch.float32)).view((1, self.dim))
                    currLoss = loss(cos(pred_x1, pred_x2), torch.tensor(y))
                    currLoss.backward()
                    optimizer.step()
                    y_pred= np.append(y_pred, cos(pred_x1, pred_x2).item())
                    #y_pred.append(cos(pred_x1, pred_x2).item())
                # y_pred = np.array(y_pred)
                rho, pval = stats.spearmanr(y_pred, train_y)
                # print("Train rho, pvalue:")
                # print(str(rho) + ', ' + str(pval))
            with torch.no_grad():
                y_pred= np.array(())
                orig=np.array(())
                for curr in test_X:
                    # x_1 = curr[:self.dim]
                    # x_2 = curr[self.dim:]
                    x_1=torch.tensor(curr[:self.dim], dtype=torch.float32)
                    x_2= torch.tensor(curr[self.dim:], dtype=torch.float32)
                    pred_x1= model(x_1).view((1, self.dim))
                    pred_x2 = model(x_2).view((1, self.dim))
                    y_pred = np.append(y_pred, cos(pred_x1, pred_x2).item())
                    # y_pred.append(cos(pred_x1, pred_x2).item())
                    orig = np.append(orig, cos(x_1.view(1, self.dim), x_2.view(1, self.dim)).item())
                    # orig.append(cos(x_1.view(1, self.dim), x_2.view(1, self.dim)).item())
                # y_pred=np.array(y_pred)
                # orig= np.array(orig)
                rho, pval = stats.spearmanr(y_pred, test_y)
                # print("Test rho, pvalue:")
                # print(str(rho)+', '+str(pval))
                test_rho=np.append(test_rho, rho)
                test_p= np.append(test_p, pval)
                rho, pval = stats.spearmanr(orig, test_y)
                init_rho = np.append(init_rho, rho)
                init_p = np.append(init_p, pval)
                # print("Initial rho, pvalue:")
                # print(str(rho) + ', ' + str(pval))
                ###Store weights
                ##Produce new embeddings in test set???
        print("Initial rho, pvalue: " +str(init_rho.mean())+ ' '+ str(init_p.mean()))
        print("Test rho, pvalue: " + str(test_rho.mean()) + ' ' + str(test_p.mean()))

def main():
    parser = argparse.ArgumentParser(description='Fitting Word Similarity')
    parser.add_argument('--dataset', type=str, default='../data/word-sim')
    parser.add_argument('--dataFile', type=str, default=None)
    parser.add_argument('--embedFile', type=str)
    parser.add_argument('--split', type=int, default=0)
    args = parser.parse_args()

    embedFile= args.embedFile
    dataFile= args.dataFile
    split= args.split
    if dataFile!= None: datasets=[dataFile]
    else: datasets= set(os.listdir(args.dataset))- set(['.DS_Store'])

    print("Processing datasets: "+ str(datasets))
    similarity_estimator= Similarity(embedFile)
    if split==0: similarity_estimator.neuralSimilarity(datasets, 5)
    else:
        for d in datasets:
            print(d)
            #similarity_estimator.run_Regressions([d])
            similarity_estimator.neuralSimilarity([d], 5)


if __name__ == '__main__':
    main()