import ast
import corenlp
import os
import pdb
import argparse
import requests
import random
from nltk.corpus import stopwords

class ConceptNetData:
    def __init__(self, corenlpPath):
        #self.rels=['IsA', 'HasProperty', 'UsedFor', 'PartOf', 'Causes', 'DefinedAs', 'MadeOf']
        ##Change them to:
        self.rels = ['IsA', 'UsedFor', 'PartOf', 'HasA', 'CreatedBy', 'MadeOf']
        self.keyPhrases = {}
        os.environ['CORENLP_HOME'] = corenlpPath
        self.CoreNLPclient = corenlp.CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'depparse', 'lemma', 'parse'])


    def construct_cn_relations(self, cn_file):
        stop_words = set(stopwords.words('english'))
        f=open(cn_file)
        dic= ast.literal_eval(f.read())
        f.close()
        relation_dic={}
        for rel in self.rels:
            for item in dic[rel]:
                cell = ast.literal_eval(item.split('\t')[4])
                entity1 = cell['surfaceStart'].lower()
                entity1=str.join(' ',list(set(entity1.split(' '))-stop_words))
                entity2 = cell['surfaceEnd'].lower()
                entity2 = str.join(' ', list(set(entity2.split(' ')) - stop_words))
                if entity1 not in relation_dic:
                    relation_dic[entity1] = {i: [] for i in self.rels}
                relation_dic[entity1][rel].append(entity2)
        refined_rels={}
        for term in relation_dic:
            frame = relation_dic[term]
            i=0
            for rel in frame:
                if len(frame[rel])>0: i+=1
            if i>1:
                refined_rels[term]=frame
        f=open('../data/ConceptNet/ConceptNet_relations.txt', 'w')
        f.write(str(refined_rels))
        f.close()



    def augment_cn_data(self, data_file):
        f=open(data_file)
        new_data= ast.literal_eval(f.read())
        f.close()
        f = open('../data/ConceptNet/' + 'conceptNet2.ibo')
        data=f.read()
        f.close()
        y_vector={}
        for rel in new_data:
            print(rel)
            print(len(new_data[rel]))
            for item in new_data[rel]:
                cell = ast.literal_eval(item.split('\t')[4])
                try:
                    text = cell['surfaceText']
                    text= text.strip('*')
                    entity1 = cell['surfaceStart']
                    entity2 = cell['surfaceEnd']
                    if '[' in entity1 or '[' in entity2:
                        text=""
                    text = str(text).replace('[', '')
                    text = text.replace(']', '')
                    span1 = (text.index(entity1), text.index(entity1) + len(entity1))
                    i = span1[0] + 1
                    span2 = (text[i:].index(entity2) + i, text[i:].index(entity2) + len(entity2) + i)
                    ann_corenlp = self.CoreNLPclient.annotate(text[0].upper() + text[1:])
                    tokens = ann_corenlp.sentence[0].token
                    deps = ann_corenlp.sentence[0].enhancedPlusPlusDependencies
                    dep_dic = {deps.root[0] - 1: {'source': -1, 'dep': 'root'}}
                    for edge in deps.edge:
                        dep_dic[edge.target - 1] = {'source': edge.source - 1, 'dep': edge.dep}
                    ibo = 'O'
                    key_ibo='O'
                    sentence = 'S' + str(len(self.keyPhrases) + 1)
                    self.keyPhrases[sentence] = [0]
                    parse= ann_corenlp.sentence[0].binarizedParseTree
                    chunkVector= self.getChunks(parse)
                    for i in range(len(tokens)):
                        token=tokens[i]
                        index = token.tokenBeginIndex
                        ##Also changed the ibo to use only span 2 as keyword
                        #SO prediction is actually over the span2 and relation (given span1???
                        #One more column is thus necessary....???Or not???
                        ibo, phrase_index = self.overlap([span2], (token.beginChar, token.endChar), ibo,
                                                         sentence)
                        key_ibo, _ = self.overlap([span1], (token.beginChar, token.endChar), key_ibo,
                                                         sentence)
                        tag='O'
                        if ibo!='O':
                            tag= ibo+rel
                        # elif ibo1!='O':
                        if key_ibo!= 'O':
                            args = [str(i + 1), token.word, token.pos, chunkVector[i], '1', tag]
                        else:
                            args = [str(i + 1), token.word, token.pos, chunkVector[i], '0', tag]
                        datum = str.join(' ', args)
                        data += datum + '\n'
                    data += '\n'
                    if span1[0] < span2[0]:
                        k = self.keyPhrases[sentence]
                        y_vector[sentence] = [rel, sentence + '.' + str(k[0]), sentence + '.' + str(k[1]), '']
                    else:
                        y_vector[sentence] = [rel, sentence + '.' + str(k[1]), sentence + '.' + str(k[0]), ',REVERSE']
                except:
                    print('Error')
                    print(item)

        f = open('../data/ConceptNet/' + 'conceptNet2.ibo', 'w')
        f.write(data)
        f.close()
        return data

    def DFS(self, curr):
        output.append(curr.value)
        if len(curr.child) == 0:
            output.append('word')
            return curr.value
        elif len(curr.child) == 1:
            self.DFS(curr.child[0])
        else:
            self.DFS(curr.child[0])
            self.DFS(curr.child[1])

    def getChunks(self, parse):
        tags = ['NP', 'VP', 'PP']
        chunkVector = []
        global output
        output = []
        curr = parse.child[0]
        self.DFS(curr)
        chunk = 'O'
        for i in range(len(output)):
            tag = output[i]
            if tag in tags:
                chunk = 'I-' + tag
            if tag == 'word':
                chunkVector.append(chunk)
        return chunkVector

    def overlap(self, spans, spanCandidate, prev, sentence):
        s2, t2= spanCandidate
        for s1, t1 in spans:
            if s2>=s1 and t2<= t1:
                prevKeyPhrase= self.keyPhrases[sentence][-1]
                if prev== 'B-':
                    return 'I-', sentence+'.'+ str(prevKeyPhrase)
                elif prev== 'I-':
                    return 'I-', sentence + '.' + str(prevKeyPhrase)
                self.keyPhrases[sentence].append(prevKeyPhrase + 1)
                return 'B-', sentence+'.'+ str(prevKeyPhrase+1)
        return 'O', sentence

    def split_data(self, data):
        data=data.split('\n\n')[:-1]
        random.shuffle(data)
        size= len(data)
        train=data[:size*8//10]
        dev=data[size*8//10:size*9//10]
        test=data[size*9//10:]
        f=open('../data/ConceptNet/train.ibo', 'w')
        for datum in train:
            f.write(datum+'\n\n')
        f.close()
        f = open('../data/ConceptNet/dev.ibo', 'w')
        for datum in dev:
            f.write(datum + '\n\n')
        f.close()
        f = open('../data/ConceptNet/test.ibo', 'w')
        for datum in test:
            f.write(datum + '\n\n')
        f.close()


    # def writeData(self, trFile, testFile):
    #     f= open(self.path+ '/'+ trFile)
    #     trainTerms= ast.literal_eval(f.read())
    #     f.close()
    #     f = open(self.path+ '/'+testFile)
    #     testTerms = ast.literal_eval(f.read())
    #     f.close()
    #     f = open(self.path+ '/' + 'train_conceptNet.ibo', 'w')
    #     y_vector={}
    #     for term in trainTerms:
    #         x, y= self.constructSentenceData(term)
    #         y_vector.update(y)
    #         f.write(x)
    #     f.close()
    #     f2 = open(self.path + '/' + 'train_conceptNet_relations.txt', 'w')
    #     for sentence in y_vector:
    #         rel, e1, e2, direction= y_vector[sentence]
    #         f2.write(rel+'('+e1+','+e2+direction+')'+'\n')
    #     f2.close()
    #     f = open(self.path+ '/'+'test_conceptNet.ibo', 'w')
    #     y_vector = {}
    #     for term in testTerms:
    #         x, y = self.constructSentenceData(term)
    #         y_vector.update(y)
    #         f.write(x)
    #     f.close()
    #     f2 = open(self.path + '/' + 'test_conceptNet_relations.txt', 'w')
    #     for sentence in y_vector:
    #         rel, e1, e2, direction = y_vector[sentence]
    #         f2.write(rel + '(' + e1 + ',' + e2 + direction + ')' + '\n')
    #     f2.close()
    #     print("Files for ConceptNet Processed...")


class WikiData:

    def __init__(self, corenlpPath, dataset=None):
        self.dataset= dataset
        self.keyPhrases={}
        os.environ['CORENLP_HOME'] = corenlpPath
        self.CoreNLPclient = corenlp.CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'depparse', 'lemma', 'parse'])

    def DFS(self, curr):
        output.append(curr.value)
        if len(curr.child) == 0:
            output.append('word')
            return curr.value
        elif len(curr.child) == 1:
            self.DFS(curr.child[0])
        else:
            self.DFS(curr.child[0])
            self.DFS(curr.child[1])

    def getChunks(self, parse):
        tags = ['NP', 'VP', 'PP']
        chunkVector = []
        global output
        output = []
        curr = parse.child[0]
        self.DFS(curr)
        chunk = 'O'
        for i in range(len(output)):
            tag = output[i]
            if tag in tags:
                chunk = 'I-' + tag
            if tag == 'word':
                chunkVector.append(chunk)
        return chunkVector

    def overlap(self, spans, spanCandidate, prev, sentence):
        s2, t2= spanCandidate
        for s1, t1 in spans:
            if s2>=s1 and t2<= t1:
                prevKeyPhrase= self.keyPhrases[sentence][-1]
                if prev== 'B-':
                    return 'I-', sentence+'.'+ str(prevKeyPhrase)
                elif prev== 'I-':
                    return 'I-', sentence + '.' + str(prevKeyPhrase)
                self.keyPhrases[sentence].append(prevKeyPhrase + 1)
                return 'B-', sentence+'.'+ str(prevKeyPhrase+1)
        return 'O', sentence

    def constructSentences(self, wikiTitle, text):
        data = ""
        wikiTitle = wikiTitle.lower()
        ann_corenlp = self.CoreNLPclient.annotate(text)
        for sIndex in range(len(ann_corenlp.sentence)):
            text = str.join(' ', [ann_corenlp.sentence[sIndex].token[i].originalText for i in
                                     range(len(ann_corenlp.sentence[sIndex].token))])
            text = text.lower()
            tokens = ann_corenlp.sentence[sIndex].token
            if wikiTitle in text:
                #deps = ann_corenlp.sentence[sIndex].enhancedPlusPlusDependencies
                parse = ann_corenlp.sentence[sIndex].binarizedParseTree
                #dep_dic = {deps.root[0] - 1: {'source': -1, 'dep': 'root'}}
                #for edge in deps.edge:
                #    dep_dic[edge.target - 1] = {'source': edge.source - 1, 'dep': edge.dep}
                flag= False
                datum=''
                sentence = 'S' + str(len(self.keyPhrases) + 1)
                self.keyPhrases[sentence] = [0]
                chunkVector = self.getChunks(parse)
                for i in range(len(tokens)):
                    token = tokens[i]
                    #key_ibo, _ = self.overlap([span], (token.beginChar, token.endChar), key_ibo, sentence)
                    if token.lemma.lower() == wikiTitle or token.word.lower() == wikiTitle:
                    #if key_ibo != 'O':
                        flag= True
                        args = [str(sIndex + 1), token.word, token.pos, chunkVector[i], '1', 'O']
                    else:
                        args = [str(sIndex + 1), token.word, token.pos, chunkVector[i], '0', 'O']
                    datum += str.join(' ', args) +'\n'
                    #data1 += datum1 + '\n'
                if flag: data += datum+ '\n'
        return data

    def constructWikiData(self, outputF, wiki_file=None, wiki_dict=None):
        if wiki_dict==None:
            if wiki_file== None:
                print("Error: please add file or dictionary...")
                return
            f=open(wiki_file)
            wiki_dict = ast.literal_eval(f.read())
            f.close()
        data = ""
        unprocessed=[]
        counter=0
        for wikiTitle in wiki_dict:
            counter+=1
            if counter%50==0:
                print('Processing: '+ str(counter)+ '/'+str(len(wiki_dict)))
            text = wiki_dict[wikiTitle]
            try:
                data+= self.constructSentences(wikiTitle, text)
            except:
                print("Could not process File")
                print(wikiTitle)
                unprocessed.append(wikiTitle)
        f = open('../data/'+outputF+'.ibo', 'w')
        f.write(data)
        f.close()


def main():
    parser = argparse.ArgumentParser(description='Augment Data with Wikipedia Links')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--corenlp', type=str, default='/Users/evangeliaspiliopoulou/Desktop/stanfordCoreNLP')
    parser.add_argument('--termF', type=str, default='../data/terms_to_defs.txt')
    parser.add_argument('--mode', type=str, default='wiki')
    args = parser.parse_args()

    mode= args.mode
    corenlpPath = args.corenlp
    dataset = args.dataset
    term_file = args.termF
    if dataset=='OpenBook':
        term_file= 'data/Wikipedia/terms_to_defs_OpenBook.txt'
    elif dataset=='ARC':
        term_file='data/Wikipedia/terms_to_defs_ARC.txt'
    if mode=='wiki':
        print('Running Wikipedia')
        wiki_constructor= WikiData(corenlpPath, dataset)
        wiki_constructor.constructWikiData('FrameTerms', wiki_file=term_file)
    elif mode== 'ConceptNet':
        print('Running CN')
        cn_constructor= ConceptNetData(corenlpPath)
        #cn_constructor.construct_cn_relations('../data/ConceptNet/allConceptNet.txt')
        data= cn_constructor.augment_cn_data('../data/ConceptNet/allConceptNet.txt')
        cn_constructor.split_data(data)
    else:
        print("Not a valid Option of Data Source")

if __name__ == '__main__':
    main()
