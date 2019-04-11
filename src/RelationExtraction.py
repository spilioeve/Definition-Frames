import torch
import torch.optim as optim
import pdb
import argparse
from BiLSTM import BiLSTM
# from BiLSTM_CNN_CRF import BiRecurrentConv, BiRecurrentConvCRF
from BiLSTM_CNN import BiLSTM_CNN
import random
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as scorer
import numpy as np
import string

torch.manual_seed(1)


class RelationExtraction:

    def __init__(self, embedFile, path, modelType, embed_Dim, charDim, hidden, num_layers):
        self.embeddings= self.setEmbeddings(path + embedFile)
        self.word_vocab={'<PAD>':0}
        self.char_vocab={'<PAD>':0}
        self.chars_inverse= {0:'<PAD>'}
        self.word_inverse={0:'<PAD>'}
        self.pos_vocab={'<PAD>':0}
        self.chunk_vocab={'<PAD>':0}
        self.labels={'<PAD>':0, 'O':1}
        self.labels_inverse = {0:'<PAD>', 1:'O'}
        self.sentenceIndex={}
        self.modelType= modelType
        self.embed_Dim= embed_Dim
        self.charDim= charDim
        self.hidden= hidden
        self.num_layers= num_layers
        self.path= path
        self.Max_Char=1
        self.use_gpu = torch.cuda.is_available()


    def setEmbeddings(self, file):
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
        return embeddings

    def loadData(self, dataType, embeddings):
        f=open(self.path+'data/'+ dataType+'.ibo')
        sentences= f.read().split('\n\n')[:-1]
        f.close()
        data=[]
        for s in sentences:
            sentence = s.split('\n')
            for wordVector in sentence:
                sentNumber, word, pos, chunk, flag, y = wordVector.split(' ')
                word = word.lower()
                if len(word)>self.Max_Char: self.Max_Char= len(word)
                for i in word:
                    if i not in self.char_vocab:
                        l= len(self.char_vocab)
                        self.char_vocab[i]=l
                        self.chars_inverse[l] =i
                if word not in self.word_vocab:
                    l = len(self.word_vocab)
                    self.word_vocab[word] = [l]
                    self.word_inverse[l] = word
                if pos not in self.pos_vocab:
                    l = len(self.pos_vocab)
                    self.pos_vocab[pos] = l
                if chunk not in self.chunk_vocab:
                    l = len(self.chunk_vocab)
                    self.chunk_vocab[chunk] = l
                if y not in self.labels:
                    self.labels[y] = len(self.labels)
                    self.labels_inverse[len(self.labels) - 1] = y
        for s in sentences:
            sentence = s.split('\n')
            x_vector = []
            y_vector = []
            words = []
            char_vector = []
            for wordVector in sentence:
                sentNumber, word, pos, chunk, flag, y = wordVector.split(' ')
                word = word.lower()
                words.append(word)
                if embeddings:
                    word_embedding = self.embeddings['UNK']
                    if word in self.embeddings:
                        word_embedding = self.embeddings[word]
                else:
                    word_embedding = self.word_vocab[word]
                pos_val = self.pos_vocab[pos]
                chunk_val = self.chunk_vocab[chunk]
                x_vector.append(word_embedding + [pos_val] + [chunk_val] + [int(flag)])  ##Changed
                y_vector.append(self.labels[y])
                char_vector.append([self.char_vocab[i] for i in word]+[self.char_vocab['<PAD>'] for j in range(self.Max_Char-len(word))])
            char_tensor = torch.tensor(char_vector, dtype=torch.long)
            x_tensor = torch.tensor(x_vector, dtype=torch.long)
            y_tensor = torch.tensor(y_vector, dtype=torch.long)
            self.sentenceIndex[str.join(' ', words)] = sentNumber
            if self.use_gpu:
                x_tensor = x_tensor.cuda()
                y_tensor = y_tensor.cuda()
                char_tensor = char_tensor.cuda()
            data.append((x_tensor, char_tensor, y_tensor))
        # for s in sentences:
        #     sentence= s.split('\n')
        #     x_vector=[]
        #     y_vector=[]
        #     words=[]
        #     char_vector=[]
        #     for wordVector in sentence:
        #         sentNumber, word, pos, chunk, flag, y= wordVector.split(' ')
        #         word=word.lower()
        #         words.append(word)
        #         for i in word:
        #             if i not in self.char_vocab:
        #                 l= len(self.char_vocab)
        #                 self.char_vocab[i]=l
        #                 self.chars_inverse[l] =i
        #         if embeddings:
        #             word_embedding = self.embeddings['UNK']
        #             if word in self.embeddings:
        #                 word_embedding= self.embeddings[word]
        #         else:
        #             if word in self.word_vocab:
        #                 word_embedding= self.word_vocab[word]
        #             else:
        #                 l = len(self.word_vocab)
        #                 word_embedding = [l]
        #                 self.word_vocab[word] = word_embedding
        #                 self.word_inverse[l]= word
        #         if pos in self.pos_vocab:
        #             pos_val= self.pos_vocab[pos]
        #         else:
        #             l= len(self.pos_vocab)
        #             pos_val= l
        #             self.pos_vocab[pos]= l
        #         if chunk in self.chunk_vocab:
        #             chunk_val = self.chunk_vocab[chunk]
        #         else:
        #             l = len(self.chunk_vocab)
        #             chunk_val = l
        #             self.chunk_vocab[chunk] = l
        #         x_vector.append(word_embedding+[pos_val]+ [chunk_val] +[int(flag)]) ##Changed
        #         if y in self.labels:
        #             y_vector.append(self.labels[y])
        #         else:
        #             self.labels[y]= len(self.labels)
        #             self.labels_inverse[len(self.labels)-1]= y
        #             y_vector.append(self.labels[y])
        #         char_vector.append([self.char_vocab[i] for i in word])
        #     char_tensor= torch.tensor(char_vector, dtype= torch.long)
        #     x_tensor= torch.tensor(x_vector, dtype=torch.long)
        #     y_tensor= torch.tensor(y_vector, dtype=torch.long)
        #     self.sentenceIndex[str.join(' ', words)]= sentNumber
        #     if self.use_gpu:
        #         x_tensor= x_tensor.cuda()
        #         y_tensor= y_tensor.cuda()
        #         char_tensor= char_tensor.cuda()
        #     data.append((x_tensor, char_tensor,  y_tensor))
        return data

    def batchify(self, data, batch_size, ftsDim, randomize= True):
        data= data
        batches=[]
        num_batches= len(data) //batch_size
        leftover= len(data) - num_batches*batch_size
        if randomize:
            random.shuffle(data)
        else:
            data=data+ data[:batch_size-leftover]
            num_batches+=1
        for b in range(num_batches):
            batch= data[b*batch_size:(b+1)*batch_size]
            batch= sorted(batch, key=self.maxCriterion, reverse=True)
            dim= batch[0][0].shape[0]
            char_dim= batch[0][1].shape[0]
            real_Lengths = [i[0].shape[0] for i in batch]
            x_tensor= torch.zeros((batch_size, dim, ftsDim), dtype=torch.long)
            y_tensor= torch.zeros((batch_size, dim), dtype=torch.long)
            char_tensor= torch.zeros((batch_size, char_dim, self.Max_Char), dtype=torch.long)
            for i in range(batch_size):
                x_i, char_i, y_i= batch[i]
                x_i= F.pad(x_i, (0, 0, 0, dim-x_i.shape[0]))
                y_i = F.pad(y_i, (0, dim - y_i.shape[0]))
                char_i= F.pad(char_i, (0, 0, 0, dim - char_i.shape[0]))
                x_tensor[i]= x_i
                y_tensor[i] = y_i
                char_tensor[i]=char_i
            batches.append((x_tensor, char_tensor, y_tensor, real_Lengths))
        return batches

    def maxCriterion(self, element):
        return len(element[0])

    def writeOutput(self, f, x, y, y_pred):
        for sentence in range(y.shape[0]):
            words=[self.word_inverse[i.item()] for i in x[sentence, :, 0]]
            flag=[str(i.item()) for i in x[sentence, :, 3]]
            true_labels=[self.labels_inverse[i.item()] for i in y[sentence]]
            pred_labels= [self.labels_inverse[i.item()] for i in y_pred[sentence]]
            words= [w for w in words if w!='<PAD>']
            word= words[0]
            words[0]= word[0].upper() + word[1:]
            sentNumber= self.sentenceIndex[str.join(' ', words).lower()]
            for w in range(len(words)):
                if words[w]=='<PAD>':
                    break
                word= words[w]
                # if w == 0:
                #     word= word[0].upper()+word[1:]
                f.write(str(sentNumber)+' '+word+' '+flag[w]+ ' '+ true_labels[w]+' '+pred_labels[w]+'\n')
            f.write('\n')

    def test(self, model, data, filePath= None, writeOutput=False):
        Pr_micro=0.000001
        Re_micro=0.000001
        Pr_macro= 0.000001
        Re_macro=0.000001
        if writeOutput:
            f=open(filePath, 'w')
        for x, char_seq, y, seq_length in data:
            #print('Predictions are:')
            y_pred = model(x, char_seq, seq_length)
            y_pred = torch.argmax(y_pred, 2)
            if writeOutput:
                self.writeOutput(f, x, y, y_pred)
            y_flat = y.view(y.shape[0]*y.shape[1])
            y_pred_flat= y_pred.view(y.shape[0]*y.shape[1])
            index= np.where(y_flat>1)
            index_pred= np.where(y_pred_flat>1)
            pr=scorer(y_flat[index_pred], y_pred_flat[index_pred], average='macro')[0]
            re = scorer(y_flat[index], y_pred_flat[index], average='macro')[1]
            Pr_macro += pr
            Re_macro += re
            # pr, re, f1, _ = scorer(y_flat, y_pred_flat, average='micro')
            re = scorer(y_flat[index], y_pred_flat[index], average='micro')[1]
            pr = scorer(y_flat[index_pred], y_pred_flat[index_pred], average='micro')[0]
            Pr_micro += pr
            Re_micro += re
                #pdb.set_trace()
        print("Micro PR, Re, F1")
        Pr_micro /= len(data)
        Re_micro /=len(data)
        F1_micro = (2 * Pr_micro * Re_micro) / (Pr_micro + Re_micro)
        print(Pr_micro, Re_micro, F1_micro)
        print("Macro PR, Re, F1")
        Pr_macro /= len(data)
        Re_macro /= len(data)
        F1_macro= (2 * Pr_macro * Re_macro) / (Pr_macro + Re_macro)
        print(Pr_macro, Re_macro, F1_macro)
        if writeOutput:
            f.close()
        print(len(data))
        return F1_micro, F1_macro


    def trainModel(self, num_epochs, trainDataSet, testDataSet, batch_size, modelType, outputPath, embeddings):
        testDataPath='Wikipedia/'+testDataSet+'/'+testDataSet
        train= self.loadData(trainDataSet+ '/train', embeddings)
        dev= self.loadData(trainDataSet+'/dev', embeddings)
        test= self.loadData(trainDataSet+'/test', embeddings)
        dev = self.batchify(dev, batch_size, 4)
        test = self.batchify(test, batch_size, 4, randomize=False)
        if modelType== 'BiLSTM_CNN':
            print('Model not currently available')
            model= BiLSTM_CNN(self.labels, len(self.word_vocab), len(self.pos_vocab), len(self.chunk_vocab), len(self.char_vocab), self.embed_Dim, self.charDim, self.hidden, self.num_layers, batch_size,
                              num_filters=30, kernel_size= 3)
            #(self, labels, vocab_size, pos_size, chunk_size, embedding_dim, hidden_dim, number_layers, batch_Size, char_embed_dim, char_size, num_filters, kernel_size):
        else:
            model= BiLSTM(self.labels, len(self.word_vocab), len(self.pos_vocab), len(self.chunk_vocab), self.embed_Dim, self.hidden, self.num_layers, batch_size)
        if self.use_gpu:
            model= model.cuda()
        # train.cuda()
        optimizer = optim.SGD(model.parameters(), lr=0.03, weight_decay=1e-4)
        print('Evaluating Train Data:')
        with torch.no_grad():
            train_i = self.batchify(train, batch_size, 4)
            self.test(model, train_i, self.path+outputPath+'/train.ibo', True)
        best_f1_macro=0.0
        best_f1_micro=0.0
        for epoch in range(num_epochs):
            print('Epoch Number:')
            print(epoch)
            total_Loss=0.
            train_i= self.batchify(train, batch_size, 4)
            for x, char, y, seq_lengths in train_i:
                model.zero_grad()
                y_pred= model(x, char, seq_lengths)
                myLoss= model.loss(y_pred, y, seq_lengths)
                total_Loss+=myLoss
                myLoss.backward()
                optimizer.step()
                del myLoss
            with torch.no_grad():
                print("Loss: Train set", total_Loss)
                #self.test(model, train_i)
                print("Evaluation: Dev set")
                f1_micro, f1_macro = self.test(model, dev)
                if f1_micro < best_f1_micro:
                    print('Early Convergence!!!!')
                    pdb.set_trace()
                    break
                elif f1_macro < best_f1_macro:
                    print('Early Convergence!!!!')
                    pdb.set_trace()
                    break
                else:
                    best_f1_micro= f1_micro
                    best_f1_macro = f1_macro
        pdb.set_trace()
        with torch.no_grad():
            print('Predictions For Test Set are:')
            self.test(model, test, self.path + outputPath + '/test'+ '.ibo', True)
            if testDataSet!= 'None':
                wiki = self.loadData(testDataPath, embeddings)
                wiki = self.batchify(wiki, batch_size, 4, randomize=False)
                self.test(model, wiki, self.path + outputPath + '/wiki_'+ str(testDataSet)+ '.ibo', True)
        torch.save(model, self.path + 'models/model'+str(num_epochs)+'_'+str(modelType)+'1_'+testDataSet)

    def evaluateModel(self, num_epochs, trainDataSet, testDataSet, batch_size, modelType, outputPath, embeddings):
        train= self.loadData(trainDataSet+ '/train', embeddings)
        dev= self.loadData(trainDataSet+'/dev', embeddings)
        test= self.loadData(trainDataSet+'/test', embeddings)
        wiki= self.loadData('Wikipedia/' + testDataSet+'/'+testDataSet, embeddings)
        dev = self.batchify(dev, batch_size, 4)
        wiki = self.batchify(wiki, batch_size, 4, randomize=False)
        test = self.batchify(test, batch_size, 4, randomize=False)
        if modelType==' CNN':
            print('Model not currently available')
            #model= CNN(self.labels, len(self.word_vocab), len(self.pos_vocab), len(self.chunk_vocab), self.embed_Dim)
        else:
            model= BiLSTM(self.labels, len(self.word_vocab), len(self.pos_vocab), len(self.chunk_vocab), self.embed_Dim, self.hidden, self.num_layers, batch_size)
        model= torch.load(self.path + 'models/model'+str(num_epochs)+'_'+str(modelType))
        model.eval()
        #self.test(model, test)
        self.test(model, test, self.path + outputPath + '/test' + '.ibo', True)
        self.test(model, wiki, self.path + outputPath + '/wiki_'+testDataSet + '.ibo', True)



def main():
    parser = argparse.ArgumentParser(description='Run Relation Extraction Module on ConceptNet')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    parser.add_argument('--model', default= 'LSTM', help='the NN model')
    parser.add_argument('--embedDim', type= int, default= 200, help='Embedding Dimension')
    parser.add_argument('--charDim', type=int, default=30, help='Char embedding Dimension')
    parser.add_argument('--hiddenDim', type=int, default=100, help='Hidden Layer Dimension')
    parser.add_argument('--path', type= str, default= '', help='Path to src code')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers in NN')
    parser.add_argument('--outputPath', type=str, default='data/output')
    parser.add_argument('--testDataset', type=str, default='OpenBook')
    parser.add_argument('--mode', type=str, default= 'train')

    args = parser.parse_args()

    embedding_Dim = args.embedDim
    char_Dim= args.charDim
    modelType= args.model
    hidden = args.hiddenDim
    num_epochs = args.num_epochs
    path= args.path
    batch_size= args.batch_size
    num_layers= args.num_layers
    outputPath= args.outputPath
    testDataset= args.testDataset
    mode= args.mode
    testDataset='None'

    rel_extractor = RelationExtraction('data/glove.6B.100d.txt', path, modelType, embedding_Dim, char_Dim, hidden, num_layers)
    if mode== 'train':
        rel_extractor.trainModel(num_epochs, 'ConceptNet', testDataset, batch_size, modelType, outputPath, embeddings=False,)
    else:
        print('Evaluating Model')
        rel_extractor.evaluateModel(num_epochs, 'ConceptNet', testDataset, batch_size, modelType,
                                 outputPath, embeddings=False)



if __name__ == '__main__':
    main()