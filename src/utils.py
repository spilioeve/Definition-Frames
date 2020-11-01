import string
from nltk.corpus import stopwords
import ast
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec
import numpy as np
from numpy.linalg import norm
from bert_embedding import BertEmbedding
import torch
import json
import pdb
from nltk.stem import WordNetLemmatizer

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

def saveBERT(embedding_file, data_file='../data/FrameTerms_refined.ibo'):
    bert = BertEmbedding(max_seq_length=100)
    f = open(data_file)
    data = f.read().strip('\n\n')
    data = data.split('\n\n')
    f.close()
    sentences=[]
    ids={}
    for sentence in data:
        words=[]
        terms= sentence.split('\n')
        if len(terms)<2:
            continue
        for wordVector in terms:
            word = wordVector.split(' ')[1]
            words.append(word)
        text=str.join(' ', words)
        if len(text)>0:
            sentences.append(text)
            ids[len(ids)]= text
    print("Data loaded...")
    #bert_encoding = bert(sentences)
    print("Data processed...")
    bert_embeddings={}
    #f = open(embedding_file, 'w')
    f = open(embedding_file)
    prev = f.readlines()
    f.close()
    for i in range(0, len(ids), 50):
        bert_encoding = bert(sentences[i:i+50])
        print(str(i)+'/'+str(len(ids)))
        for j in range(50):
            #id= ids[i+j]
            #bert_embed= bert(sentences[i])[0][1]
            if j<len(bert_encoding):
                bert_embed= bert_encoding[j][1]
                x_tensor = torch.tensor(bert_embed, dtype=torch.float)
                vector = x_tensor.tolist()
                #f.write(id+'\t'+vector+'\n')
                bert_embeddings[i+j]= vector
    f= open(embedding_file, 'w')
    for line in prev:
        f.write(line)
    for i in range(len(ids)):
        f.write(str(bert_embeddings[i])+'\n')
    f.close()
    f=open('../data/embeddings/bert/indexer.json')
    indexer=json.load(f)
    f = open('../data/embeddings/bert/indexer2.json', 'w')
    for i in range(len(indexer)):
        f.write(indexer[i]+'\n')
    for s in sentences:
        f.write(s+'\n')
    f.close()
    # pdb.set_trace()
    # np.save(embedding_file, bert_embeddings)
    # f=open('bert_indexer2.json', 'w')
    # json.dump(ids, f)


#for line in lines:
#     items= line.split('\t')
#     x1= items[5]
#     out1= ann.annotate(x1).sentence[0].token
#     for token in out1:
#         lemma=t.lemma
#         pos= t.pos
#         if "NN" in pos or "VB" in pos:
#             vocab.add(lemma)
#     x2= items[6]
#     out1= ann.annotate(x2).sentence[0].token
#     for t in out1:
#         lemma=t.lemma
#         pos = t.pos
#         if "NN" in pos or "VB" in pos:
#             vocab.add(lemma)

# def classes(d, frame_terms, size):
#     all_classes = {i: [0.0, 0.0, 0.0] for i in range(size)}
#     for i in range(4906):
#         #d = data[i].split('\t')
#         count = 0
#         text = d[0] + ' ' + d[1]
#         for word in text.split():
#             word= word.lower()
#             if word in frame_terms:
#                 count += 1
#         all_classes[count][0] += (d[2] - d[4]) ** 2
#         all_classes[count][1] += (d[3] - d[4]) ** 2
#         all_classes[count][2] += 1
#     for i in all_classes:
#         if all_classes[i][2]!= 0.0:
#             all_classes[i][0] /= all_classes[i][2]
#             all_classes[i][1] /= all_classes[i][2]
#     return all_classes
#
# def sent_class(local_data, frame_terms, size):
#     all_classes = {i: [] for i in range(size)}
#     for i in range(len(local_data)):
#         d = local_data[i].split('\t')
#         count = 0
#         text = d[0] + ' ' + d[1]
#         for word in text.split():
#             word= word.lower()
#             if word in frame_terms:
#                 count += 1
#         #d[2]= float(round(Decimal(d[2]), 4))
#         #d[3] = float(round(Decimal(d[3]), 4))
#         d[2]= float(d[2])
#         d[3]= float(d[3])
#         d[4]= float(d[4])
#         d[5]= count
#         all_classes[count].append(d)
#     return all_classes

saveBERT('../data/embeddings/bert/all_embeddings.txt')