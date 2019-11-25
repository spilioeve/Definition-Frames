import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pdb
import argparse
import os
relations = ['IsA', 'UsedFor', 'PartOf', 'MadeOf', 'HasA']

def readEmbeddings(file, dimension):
    f=open('../frame_embeddings/'+file)
    lines=f.read().split('\n')[:-1]
    f.close()
    dim2 = int(len(lines[0].split()[1:]) / dimension)
    tensors=np.zeros((0, dim2, dimension))
    vocab=[]
    for l in lines:
        items=l.split()
        vector=[float(i) for i in items[1:]]
        tensor_i =np.reshape(np.array(vector), (1, dim2, dimension))
        vocab.append(items[0])
        tensors = np.vstack((tensors, tensor_i))
    return tensors, vocab

def dim_reduction(num_components, tensors, modelType='pca'):
    transformed_tensors= np.zeros((tensors.shape[0], tensors.shape[1], num_components))
    for dimension in range(tensors.shape[1]):
        #pca = PCA(n_components= num_components)
        if modelType == 'pca':
            model = PCA(n_components=num_components)
        elif modelType == 'tsne':
            model = TSNE(n_components=num_components)
        trans_tensor= model.fit_transform(tensors[:, dimension, :])
        transformed_tensors[:, dimension, :] = trans_tensor
    return transformed_tensors

def plotVectors(tensors, vocab, init_tensors, all_pairs= {}, score=0.7):
    words=set()
    if len(all_pairs)==0: words= set(vocab)
    else:
        for (w1, w2) in all_pairs:
            #pairs=all_pairs[d]
            #for (w1, w2) in pairs:
            if all_pairs[(w1, w2)]> score and w1 in vocab and w2 in vocab:
                words.add(w1)
                words.add(w2)
    zeros=np.zeros(init_tensors.shape[2])
    counter=0
    for dimension in range(1, tensors.shape[1]):
        relation= relations[dimension-1]
        tensors_dim= tensors[:, dimension, :]
        #plt.plot(tensors_dim[:, 0], tensors_dim[:, 1], 'co')
        for word in words:
        #['rain', 'bee', 'signature']
        #for index in range(len(vocab)):
            #word= vocab[index]
            index=vocab.index(word)
            initial= init_tensors[index, dimension, :]
            if not np.array_equal(initial, zeros):
                counter+=1
                plt.plot(tensors_dim[index, 0], tensors_dim[index, 1], 'co')
                plt.annotate(word, (tensors_dim[index, 0], tensors_dim[index, 1]))
        plt.savefig('../figures/Frames_Sim_pca2_'+relation+'.pdf')
        plt.clf()
        plt.cla()
        plt.close()
        print("Number of points for " + str(relation))
        print(counter)
    #####
    # for pair in pairs:
    #     w1, w2, score= pair
    #     if score>0.9:
    #         pass

def loadData(datasets):
    pairs = {}
    for d in datasets:
        f = open('../data/word-sim/' + d)
        lines = f.read().split('\n')[:-1]
        f.close()
        norm = 0.0
        data_pairs = {}
        for line in lines:
            word1, word2, score = line.split()
            score = float(score)
            if score > norm: norm = int(score+0.99)
            data_pairs[(word1, word2)] = score
        pairs.update({i: data_pairs[i] / norm for i in data_pairs})
    #print('Number of pairs found: ' + str(pairCount) + '/' + str(len(pairs)))
    return pairs

def main():
    parser = argparse.ArgumentParser(description='Fitting Word Similarity')
    parser.add_argument('--dataset', type=str, default='../data/word-sim')
    parser.add_argument('--dataFile', type=str, default=None)
    #parser.add_argument('--embedFile', type=str)
    args = parser.parse_args()

    dataFile = args.dataFile
    if dataFile!= None: datasets=[dataFile]
    else: datasets= set(os.listdir(args.dataset))- set(['.DS_Store'])

    tensors, vocab= readEmbeddings('Frames_Cglove.txt', 100)
    print('Tensors read')

    transformed= dim_reduction(2, tensors)

    print('Tensors transformed')
    all_pairs= loadData(datasets)
    ###Load the pairs to get LESS vectors, seems so fucking complicated this graph! But this is only for plotting, NOT actual
    #transform
    plotVectors(transformed, vocab, tensors, all_pairs= all_pairs, score=0.8)

if __name__ == '__main__':
    main()