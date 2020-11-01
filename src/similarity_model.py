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



def load_data(path, datasets, embeddings, dim, embeddings_base=None, dim_base= None):
    pairCount=0.
    pairs={}
    for d in datasets:
        f=open(path+d)
        lines=f.read().split('\n')[:-1]
        f.close()
        norm=0.0
        data_pairs={}
        for line in lines:
            word1, word2, score= line.split()
            score=float(score)
            if score>norm: norm= int(score+0.99)
            data_pairs[(word1, word2)]= score
        pairs.update({i:data_pairs[i]/norm for i in data_pairs})

    X = np.zeros((2*dim))
    if embeddings_base!= None:
        X_base = np.zeros((2 * dim_base))
    y = []
    pairWords=list(pairs.keys())
    random.shuffle(pairWords)
    for word1, word2 in pairWords:
        if word1 in embeddings and word2 in embeddings:
            v1 = np.array(embeddings[word1])
            v2 = np.array(embeddings[word2])
            #x_i= np.outer(v1, v2).flatten()
            x_1 = np.concatenate((v1, v2))
            x_2 = np.concatenate((v2, v1))
            X = np.vstack((X, x_1))
            X = np.vstack((X, x_2))
            y.append(pairs[(word1, word2)])
            y.append(pairs[(word1, word2)])
            pairCount+=1
            if embeddings_base != None:
                v1_base = np.array(embeddings_base[word1])
                v2_base = np.array(embeddings_base[word2])
                x1_base = np.concatenate((v1_base, v2_base))
                x2_base = np.concatenate((v2_base, v1_base))
                X_base = np.vstack((X_base, x1_base))
                X_base = np.vstack((X_base, x2_base))
    X = X[1:, :]
    Y = np.array(y)
    print('Number of pairs found: ' + str(pairCount) + '/' + str(len(pairs)))
    if embeddings_base!= None:
        X_base = X_base[1:, :]
        return X, X_base, Y
    return X, Y

def read_embeddings(embedding_file):
    f=open('../frame_embeddings/'+embedding_file)
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

def run_regression(datasets, embeddings, dim):
    X, Y= load_data(datasets, embeddings, dim)
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



#Maybe I should shuffle it so instead of giving weight to all dimensions, I give weight to ONLY a vector??? This would be much easier,
#and pretty sure that GloVe would suck then
def cross_validation(path, datasets, epochs, embedding_file, embedding_base_file):
    embeddings, dim = read_embeddings(embedding_file)
    embeddings_base, dim_base = read_embeddings(embedding_base_file)
    X, X_base, Y = load_data(path, datasets, embeddings, dim, embeddings_base=embeddings_base, dim_base= dim_base)
    #X_, Y_ =self.loadBaseline(datasets)
    num_data = X.shape[0]
    kf = KFold(n_splits=10)
    if num_data < 100: return
    for x, dimension in [(X, dim), (X_base, dim_base)]:
        rho_test = np.array(())
        p_test = np.array(())
        rho_init = np.array(())
        p_init = np.array(())
        for train, test in kf.split(x):
            train_X = x[train]
            train_y= Y[train]
            test_X= x[test]
            test_y=Y[test]
            model= nn.Linear(dimension, dimension)
            cos = nn.CosineSimilarity()
            loss = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.07, weight_decay=1e-4)
            #optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-4)
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
                    x_1= train_X[i][:dimension]
                    x_2 = train_X[i][dimension:]
                    y= train_y[i]
                    pred_x1= model(torch.tensor(x_1, dtype=torch.float32)).view((1, dimension))
                    pred_x2 = model(torch.tensor(x_2, dtype=torch.float32)).view((1, dimension))
                    currLoss = loss(cos(pred_x1, pred_x2), torch.tensor(y))
                    currLoss.backward()
                    optimizer.step()
                    y_pred= np.append(y_pred, cos(pred_x1, pred_x2).item())
            with torch.no_grad():
                y_pred= np.array(())
                y_init=np.array(())
                for curr in test_X:
                    x_1=torch.tensor(curr[:dimension], dtype=torch.float32)
                    x_2= torch.tensor(curr[dimension:], dtype=torch.float32)
                    pred_x1= model(x_1).view((1, dim))
                    pred_x2 = model(x_2).view((1, dim))
                    y_pred = np.append(y_pred, cos(pred_x1, pred_x2).item())
                    y_init = np.append(y_init, cos(x_1.view(1, dimension), x_2.view(1, dimension)).item())
                rho, pval = stats.spearmanr(y_pred, test_y)
                rho_test=np.append(rho_test, rho)
                p_test= np.append(p_test, pval)
                rho, pval = stats.spearmanr(y_init, test_y)
                rho_init = np.append(rho_init, rho)
                p_init = np.append(p_init, pval)
        print("Definition Frames / Base embeddings:  ")
        print("Initial rho, pvalue: " +str(rho_init.mean())+ ' '+ str(p_init.mean()))
        print("Test rho, pvalue: " + str(rho_test.mean()) + ' ' + str(p_test.mean()))


def train(path, train_dataset, test_datasets, epochs, embedding_file):
    embeddings, dim = read_embeddings(embedding_file)
    train_x, train_y = load_data(path, train_dataset, embeddings, dim)
    #X_, Y_ =self.loadBaseline(datasets)

    model = nn.Linear(dim, dim)
    cos = nn.CosineSimilarity()
    loss = nn.MSELoss()

    optimizer = optim.SGD(model.parameters(), lr=0.07, weight_decay=1e-4)
    for epoch in range(epochs):
        y_pred = np.array(())
        randomize = np.arange(len(train_y))
        np.random.shuffle(randomize)
        train_x = train_x[randomize]
        train_y = train_y[randomize]
        for i in range(len(train_x)):
            model.zero_grad()
            x1 = train_x[i][:dim]
            x2 = train_x[i][dim:]
            y = train_y[i]
            x1_pred= model(torch.tensor(x1, dtype=torch.float32)).view((1, dim))
            x2_pred = model(torch.tensor(x2, dtype=torch.float32)).view((1, dim))
            loss_curr = loss(cos(x1_pred, x2_pred), torch.tensor(y))
            loss_curr.backward()
            optimizer.step()
            y_pred= np.append(y_pred, cos(x1_pred, x2_pred).item())
        #rho, pval = stats.spearmanr(y_pred, train_y)
    for dataset in test_datasets:
        test_x, test_y = load_data(path, [dataset], embeddings, dim)
        rho_test = np.array(())
        p_test = np.array(())
        rho_init = np.array(())
        p_init = np.array(())
        with torch.no_grad():
            y_pred = np.array(())
            y_init = np.array(())
            for i in range(len(test_x)):
                x1=torch.tensor(test_x[i][:dim], dtype=torch.float32)
                x2= torch.tensor(test_x[i][dim:], dtype=torch.float32)
                x1_pred= model(x1).view((1, dim))
                x2_pred = model(x2).view((1, dim))
                y_pred = np.append(y_pred, cos(x1_pred, x2_pred).item())
                y_init = np.append(y_init, cos(x1.view(1, dim), x2.view(1, dim)).item())
            rho, pval = stats.spearmanr(y_pred, test_y)
            rho_test=np.append(rho_test, rho)
            p_test= np.append(p_test, pval)
            rho, pval = stats.spearmanr(y_init, test_y)
            rho_init = np.append(rho_init, rho)
            p_init = np.append(p_init, pval)
        print('Dataset: '+ str(dataset))
        print("Initial rho, pvalue: " +str(rho_init.mean())+ ' '+ str(p_init.mean()))
        print("Test rho, pvalue: " + str(rho_test.mean()) + ' ' + str(p_test.mean()))

def main():
    parser = argparse.ArgumentParser(description='Fitting Word Similarity')
    parser.add_argument('--dataset', type=str, default='../data/word-sim')
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--train_dataset', type=str, default=None)
    parser.add_argument('--embedding_file', type=str)
    parser.add_argument('--embedding_base_file', type=str, default=None)
    parser.add_argument('--embedding_base_file2', type=str, default=None)
    parser.add_argument('--mode', type=str, default="train")
    args = parser.parse_args()

    embedding_file= args.embedding_file
    embedding_base_file= args.embedding_base_file
    embedding_base_file2 = args.embedding_base_file2
    data_file= args.data_file
    mode= args.mode
    dataset= args.dataset

    if data_file!= None: datasets=[data_file]
    else: datasets= set(os.listdir(dataset))- set(['.DS_Store'])

    print("Processing datasets: "+ str(datasets))

    if mode=="train":
        train_dataset= args.train_dataset
        test_datasets= datasets- set(train_dataset)
        print("Training on: " + str(train_dataset))
        train(dataset+'/', [train_dataset], test_datasets, 15, embedding_file)
    else:

        cross_validation(dataset+'/', datasets, 10, embedding_file, embedding_base_file)
        for d in datasets:
            print(d)
            #similarity_estimator.run_Regressions([d])
            cross_validation(dataset+'/', [d], 10, embedding_file, embedding_base_file)


if __name__ == '__main__':
    main()
