import string
from nltk.corpus import stopwords
import ast
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import numpy as np
from numpy.linalg import norm

def cleanTokens(text):
    stop_words = set(stopwords.words('english'))
    stop_words.add('-lrb-')
    stop_words.add('-rrb-')
    punct = string.punctuation
    p = [i for i in punct]
    punct = set(p)
    for i in p: punct.add(i + i)
    words=text.split(' ')
    new_text=''
    for word in words:
        if word not in punct:
            if word not in stop_words:
                new_text+= word+ ' '
    new_text= new_text.strip(' ')
    return new_text

def setEmbeddings(embed_type):
    file = '../data/glove.6B.100d.txt'
    if embed_type== 'glove':
        file= '../data/glove.6B.100d.txt'
        embeddings = {}
        f=open(file)
        lines=f.read().split('\n')[:-1]
        f.close()
        for line in lines:
            vector= line.split(' ')
            word= vector[0]
            vector= [float(i) for i in vector[1:]]
            embeddings[word]= vector
        embeddings['UNK'] = len(vector) * [0.0]
    elif embed_type=='word2vec':
        f = open('../data/terms_to_defs.txt')
        termDic = ast.literal_eval(f.read())
        f.close()
        text = ""
        for term in termDic:
            text += termDic[term]
        data=[]
        for s in sent_tokenize(text):
            words=[]
            for w in word_tokenize(s):
                words.append(w.lower())
            data.append(words)
        w2v_model= gensim.models.Word2Vec(data, min_count = 1,size = 100, window = 5)
        return w2v_model, 100
    elif embed_type=='model_embeds':
        file = '../frame_embeddings/model_embeds.txt'
        embeddings = {}
        f = open(file)
        lines = f.read().split('\n')[:-1]
        f.close()
        for line in lines:
            vector = line.split(' ')
            word = vector[0]
            vector = [float(i) for i in vector[1:]]
            embeddings[word] = vector
        embeddings['UNK'] = len(vector) * [0.0]
    else:
        w2v_model= Word2Vec.load('../data/word2vec/'+embed_type)
        return w2v_model, 100
    return embeddings, len(vector)


def cos_sparse_sim(vector1, vector2, dim):
    embedDim, relDim= dim
    vector1= np.reshape(np.array(vector1), (relDim, embedDim))
    vector2 = np.reshape(np.array(vector2), (relDim, embedDim))
    vec1= vector1[~np.all(vector1 == 0, axis=1)].flatten()
    vec2= vector2[~np.all(vector1 == 0, axis=1)].flatten()
    return vec1.dot(vec2)/(norm(vec1)*norm(vec2))


