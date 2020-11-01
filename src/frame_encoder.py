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

def setEmbeddings(embedding_type):
    #file = '../data/embeddings/glove.6B.50d.txt'
    embedding_file = '../data/embeddings/'+ embedding_type+'.txt'
    if embedding_type== 'glove':
        embedding_file = '../data/embeddings/glove.6B.100d.txt'
    ##Didn't work, erase it
    elif embedding_type=='word2vec':
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
    ##Didn't work, erase it
    elif embedding_type=='model_embeds':
        embedding_file = '../frame_embeddings/model_embeds.txt'
    elif embedding_type =='dict2vec':
        embedding_file = '../data/embeddings/dict2vec-vectors-dim100.vec'
    embeddings = {}
    f=open(embedding_file)
    lines=f.read().split('\n')[:-1]
    f.close()
    for line in lines:
        vector= line.split(' ')
        word= vector[0]
        if vector[-1]=='': vector=vector[:-1]
        vector= [float(i) for i in vector[1:]]
        embeddings[word]= vector
    embeddings['UNK'] = len(vector) * [0.0]
    return embeddings, len(vector)

class Frame2Vec:
    def __init__(self, term_list, def_file, relations, embedding_type):
        self.term_list= term_list
        self.relations = relations
        self.lmtzr = WordNetLemmatizer()
        print('Number of total Terms:')
        print(len(term_list))
        self.frames= self.read_frames('../data/output/'+def_file)
        self.embeddings, dimension = setEmbeddings(embedding_type)
        self.dimension = (dimension, len(self.relations))

    def read_frames(self, def_file):
        f=open('../data/Frames4Terms2.txt', 'w')
        frame_relations= self.extract_relations(def_file)
        all_frames={}
        counter=0
        for term in self.term_list:
            term=term.lower()
            lemma=self.lmtzr.lemmatize(term)
            if lemma in frame_relations:
                counter+=1
                frames={r:[] for r in self.relations}
                for relations, sentNumber in frame_relations[lemma]:
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

    def frameAvgEmbeddings(self, output_file, embedding_type):
        f=open('../frame_embeddings/'+output_file, 'w')
        #f1 = open('../frame_embeddings/average_w2v_wn.txt', 'w')
        f2 = open('../frame_embeddings/'+embedding_type+'.txt', 'w')
        for term in self.term_list:
            term = term.lower()
            if term in self.frames and term in self.embeddings:
                embedding = list(self.embeddings[term][:])
                rel_terms=[term]
                frame_embedding = embedding[:]
                frame= self.frames[term]
                for relation in self.relations:
                    words= frame[relation]
                    rel_terms+= words
                    vector= self.dimension[0]*[0.]
                    if len(words)>0:
                        vector= self.phrase_embeddings(words)
                    frame_embedding+= vector
                frame_embedding = str.join(' ', [str(i) for i in frame_embedding])
                f.write(term + ' ' + frame_embedding + '\n')
                embedding= embedding+self.dimension[0]*self.dimension[1]*[0.]
                embedding = str.join(' ', [str(i) for i in embedding])
                f2.write(term+' '+embedding+'\n')
        f.close()
        f2.close()

    def phrase_embeddings(self, terms):
        avg_embedding= self.dimension[0]*[0.]
        count = 0
        for term in terms:
            if term in self.embeddings:
                embedding = list(self.embeddings[term][:])
                avg_embedding = [x + y for x, y in zip(avg_embedding, embedding)]
                count += 1
        if count > 1:
            avg_embedding= [i/float(count) for i in avg_embedding]
        return avg_embedding


    def extract_relations(self, def_file):
        f= open(def_file)
        output= f.read().split('\n\n')[:-1]
        f.close()
        keyTerms={}
        parsed_sentences=set()
        print(len(output))
        print("processing sentences...")
        for sentence in output:
            if len(sentence)<1:
                continue
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
                        flag= terms[4]
                        relation = terms[6]
                        if flag=='1' and len(key)==0: key = self.lmtzr.lemmatize(word)
                        if relation != 'O':
                            if relation not in relations:
                                    relations[relation]= []
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
    parser.add_argument('--def_file', type=str, default='FrameTerms_wn_ann.ibo')
    parser.add_argument('--term_file', type=str, default='../data/word-sim/ws_terms_nominals')
    parser.add_argument('--output_file', type=str, default='CN_Frames_w2v.txt')
    parser.add_argument('--embedding_type', type=str, default='glove')
    parser.add_argument('--relations', type=str, default='IsA,UsedFor,PartOf,MadeOf,HasA,CreatedBy')

    args = parser.parse_args()
    output_file=args.output_file
    def_file=args.def_file
    embedding_type=args.embedding_type
    relations= args.relations.split(',')
    f = open(args.term_file)
    term_list = f.read().split('\n')[:-1]
    f.close()

    encoder=Frame2Vec(term_list, def_file, relations, embedding_type)
    print("Encoded")
    encoder.frameAvgEmbeddings(output_file, embedding_type)

if __name__ == '__main__':
    main()