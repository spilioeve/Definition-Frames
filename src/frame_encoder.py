import nltk
import string
from nltk.corpus import stopwords
import ast

stop_words=set(stopwords.words('english'))
stop_words.add('-lrb-')
stop_words.add('-rrb')
punc=string.punctuation

def cleanTokens(text):
    words=text.split(' ')
    new_text=''
    for word in words:
        if word not in punc:
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
    return embeddings

class Frame2Vec:
    def __init__(self, termList, framePath, embeddingType):
        self.embeddings, dimension= setEmbeddings(embeddingType)
        self.termList= termList
        frames= self.readFrames(framePath)
        self.relations=['is_a', 'defined_as', 'used_for','']
        self.dimension= dimension



    def readFrames(self, framePath):
        f=open(framePath)
        all_frames=ast.literal_eval(f.read())
        f.close()
        frames={}
        for term in self.termList:
            text_frame= all_frames[term]
            frame={}
            for key in text_frame:
                text= text_frame[key]
                tokens= cleanTokens(text)
                frame[key]= tokens
            frames[term]= frame
        return frames


    def frameAvgEmbeddings(self, frames, outputF):
        f=open(outputF)
        for term in frames:
            frame= frames[term]
            frame_embedding=self.dimension*[0.0]
            if term in self.embeddings:
                frame_embedding=self.embeddings[term]
            for dimension in self.relations:
                vector= self.dimension* [0.]
                if dimension in frame:
                    text=frame[dimension]
                    vector= self.getAvgEmbeddings(text)
                frame_embedding+= vector
            frame_embedding= str.join(' ', [str(i) for i in frame_embedding])
            f.write(term + ' ' + frame_embedding+'\n')
        f.close()

    def getAvgEmbeddings(self, phrase):
        avgEmbed = self.embeddings['UNK'][:]
        count = 0
        for term in phrase.split(' '):
            if term in self.embeddings:
                embedding = self.embeddings[term][:]
                avgEmbed = [x + y for x, y in zip(avgEmbed, embedding)]
                count += 1
        if count > 1:
            avgEmbed= [i/float(count) for i in avgEmbed]
        return avgEmbed

