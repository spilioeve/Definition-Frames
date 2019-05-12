import nltk
import string
from nltk.corpus import stopwords
import ast
from nltk.stem import WordNetLemmatizer
import pdb
import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec

stop_words=set(stopwords.words('english'))
stop_words.add('-lrb-')
stop_words.add('-rrb-')
punct=string.punctuation
p=[i for i in punct]
punct=set(p)
for i in p: punct.add(i+i)

def cleanTokens(text):
    words=text.split(' ')
    new_text=''
    for word in words:
        if word not in punct:
            if word not in stop_words:
                new_text+= word+ ' '
    new_text= new_text.strip(' ')
    return new_text

def setEmbeddings(embed_type):
    #file = '../data/embeddings/glove.6B.50d.txt'
    if embed_type== 'glove':
        embeddings = {}
        f=open('../data/embeddings/glove.6B.50d.txt')
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
    elif embed_type =='dict2vec':
        file = '../data/embeddings/dict2vec-vectors-dim100.vec'
        embeddings = {}
        f = open(file)
        lines = f.read().split('\n')[1:-1]
        f.close()
        for line in lines:
            vector = line.split(' ')
            word = vector[0]
            vector = [float(i) for i in vector[1:-1]]
            embeddings[word] = vector
        embeddings['UNK'] = len(vector) * [0.0]
    else:
        w2v_model= Word2Vec.load('../data/embeddings/word2vec/'+embed_type)
        return w2v_model, 100
    return embeddings, len(vector)

class Frame2Vec:
    def __init__(self, termList, framePath, embeddingType, writeEmbeds=True):
        self.writeEmbeds= writeEmbeds
        self.termList= termList
        #self.relations = ['IsA', 'UsedFor', 'PartOf', 'MadeOf', 'HasA']
        #self.relations = ['IsA', 'UsedFor', 'MadeOf']
        self.relations = ['IsA']
        self.lmtzr = WordNetLemmatizer()
        print('Number of total Terms:')
        print(len(termList))
        self.frames= self.readFrames(framePath)
        #self.frames=self.readCN_Rels()
        self.embeddings, dimension = setEmbeddings(embeddingType)
        self.dimension = (dimension, len(self.relations))

    def readFrames(self, framePath):
        f=open('../data/Frames4Terms2.txt', 'w')
        frameRelations= self.outputToRelations(framePath)
        all_frames={}
        counter=0
        for term in self.termList:
            term=term.lower()
            lemma=self.lmtzr.lemmatize(term)
            if lemma in frameRelations:
                counter+=1
                frames={r:[] for r in self.relations}
                for relations, sentNumber in frameRelations[lemma]:
                    for rel in relations:
                        if rel in self.relations:
                            words= relations[rel]
                            frames[rel]+=words
                all_frames[term]=frames
        print('Number of terms in Def Frames:')
        print(counter)
        f.write(str(all_frames))
        f.close()
        return all_frames

    def readCN_Rels(self):
        f=open('../data/ConceptNet/ConceptNet_relations.txt')
        relations=ast.literal_eval(f.read())
        f.close()
        relations2={}
        frames={}
        # for item in relations:
        #     if len(item.split(' '))==1:
        #         lemma = self.lmtzr.lemmatize(item)
        #         relations2[lemma]=relations[item]
        for term in self.termList:
            lemma = self.lmtzr.lemmatize(term)
            if lemma in relations:
                frames[term]= relations[lemma]
        return frames

    def frameAvgEmbeddings(self, outputF):
        f=open(outputF, 'w')
        #f1 = open('../frame_embeddings/average_w2v_wn.txt', 'w')
        f2 = open('../frame_embeddings/dict2vec.txt', 'w')
        for term in self.termList:
            term = term.lower()
            flag=True
            # if term in self.embeddings:
            #     embedding=self.embeddings[term][:]
            # else:
            #     embedding = self.embeddings['UNK'][:]
            #     flag=False
            if term in self.frames and term in self.embeddings:
                embedding = list(self.embeddings[term][:])
                allterms=[term]
                avg_embedding=embedding[:]
                frame_embedding = embedding[:]
                #embedding= str.join(' ', [str(i) for i in embedding])
                flag = True
                frame= self.frames[term]
                for relation in self.relations:
                    words= frame[relation]
                    allterms+= words
                    #vector=embedding[:]
                    vector= self.dimension[0]*[0.]
                    if len(words)>0:
                        vector= self.getAvgEmbeddings(words)
                    frame_embedding+= vector
                frame_embedding = str.join(' ', [str(i) for i in frame_embedding])
                f.write(term + ' ' + frame_embedding + '\n')
                embedding= embedding+self.dimension[0]*self.dimension[1]*[0.]
                embedding = str.join(' ', [str(i) for i in embedding])
                f2.write(term+' '+embedding+'\n')
                # avg_embedding += self.getAvgEmbeddings(allterms)
                # avg_embedding= str.join(' ', [str(i) for i in avg_embedding])
                # f1.write(term + ' ' + avg_embedding + '\n')
            # else:
            #     frame_embedding = embedding+ self.dimension[0]*self.dimension[1] * [0.0]
            #     #frame_embedding = (self.dimension[1]+1)* embedding
            # if flag:
            #     frame_embedding = str.join(' ', [str(i) for i in frame_embedding])
            #     f.write(term + ' ' + frame_embedding+'\n')
        f.close()
        #f1.close()
        f2.close()

    def avgFrameEmbedding(self, outputF):
        f=open(outputF, 'w')
        for term in self.termList:
            term = term.lower()
            flag=True
            if term in self.embeddings:
                embedding=self.embeddings[term][:]
                if term in self.frames:
                    related_terms=[]
                    embedding = self.embeddings[term][:]
                    frame= self.frames[term]

                    for relation in self.relations:
                        words= frame[relation]
                        #vector=embedding[:]
                        vector= self.dimension[0]*[0.]
                        if len(words)>0:
                            vector= self.getAvgEmbeddings(words)
                        frame_embedding+= vector
                    frame_embedding = str.join(' ', [str(i) for i in frame_embedding])
                f.write(term + ' ' + frame_embedding + '\n')
            # else:
            #     frame_embedding = embedding+ self.dimension[0]*self.dimension[1] * [0.0]
            #     #frame_embedding = (self.dimension[1]+1)* embedding
            # if flag:
            #     frame_embedding = str.join(' ', [str(i) for i in frame_embedding])
            #     f.write(term + ' ' + frame_embedding+'\n')
        f.close()

    def getAvgEmbeddings(self, terms):
        avgEmbed= self.dimension[0]*[0.]
        # avgEmbed = self.embeddings['UNK'][:]
        count = 0
        for term in terms:
            if term in self.embeddings:
                embedding = list(self.embeddings[term][:])
                avgEmbed = [x + y for x, y in zip(avgEmbed, embedding)]
                count += 1
        if count > 1:
            avgEmbed= [i/float(count) for i in avgEmbed]
        return avgEmbed


    def outputToRelations(self, outputF):
        f= open(outputF)
        output= f.read().split('\n\n')[:-1]
        f.close()
        keyTerms={}
        currIndex=0
        parsed_sentences=set()
        print(len(output))
        print("processing sentences...")
        for sentence in output:
            if sentence not in parsed_sentences:
                parsed_sentences.add(sentence)
                key=""
                relations = {}
                lines= sentence.split('\n')
                sentNumber=1
                for item in lines:
                    terms= item.split(' ')
                    sentNumber=int(terms[0])
                    word=terms[1].lower()
                    if (word not in punct) and (word not in stop_words) and (sentNumber<3):
                        flag= terms[2]
                        relation = terms[4]
                        if flag=='1' and len(key)==0: key = self.lmtzr.lemmatize(word)
                        if relation != 'O':
                            start, relation= relation.split('-')
                            if relation not in relations:
                                    relations[relation]= [word]
                            # elif start == 'I':
                            #     prev= relations[relation][-1]
                            #     relations[relation][-1] = prev +' '+ word
                            else:
                                relations[relation].append(word)
                if key=="":
                    pass
                elif key not in keyTerms:
                    keyTerms[key] = [(relations, sentNumber)]
                else:
                    keyTerms[key].append((relations, sentNumber))
        return keyTerms


def main():
    parser = argparse.ArgumentParser(description='Getting Definition Frame for term(s)')
    parser.add_argument('--input', type=str, default='../data/output/wiki_FrameTerms.ibo')
    parser.add_argument('--termFile', type=str, default='../data/word_sim_ALL.txt')
    parser.add_argument('--output', type=str, default='../frame_embeddings/CN_Frames.txt')
    parser.add_argument('--embeds', type=str, default='glove')
    parser.add_argument('--writeEmbeds', type=int, default=0)

    args = parser.parse_args()
    termFile=args.termFile
    outputF=args.output
    inputF=args.input
    embedType=args.embeds
    writeEmbeds= args.writeEmbeds

    f = open(termFile)
    termList = f.read().split('\n')[:-1]
    f.close()

    encoder=Frame2Vec(termList, inputF, embedType, writeEmbeds)
    print("Encoded")
    encoder.frameAvgEmbeddings(outputF)

if __name__ == '__main__':
    main()