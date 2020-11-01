import corenlp
import os
import pdb
import ast
import argparse
import requests
import random
from nltk.corpus import stopwords
import json


class Data_Processor:

    def __init__(self, corenlp_path):

        self.keyPhrases={}
        self.rels = ['IsA', 'UsedFor', 'PartOf', 'HasA', 'CreatedBy', 'MadeOf']
        os.environ['CORENLP_HOME'] = corenlp_path
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

    def get_chunks(self, parse):
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

    def process_sentence(self, term, text, mode, spans= None):
        data = ""
        term = term.lower()
        ann_corenlp = self.CoreNLPclient.annotate(text)
        for sIndex in range(len(ann_corenlp.sentence)):
            text = str.join(' ', [ann_corenlp.sentence[sIndex].token[i].originalText for i in
                                     range(len(ann_corenlp.sentence[sIndex].token))])
            text = text.lower()
            tokens = ann_corenlp.sentence[sIndex].token
            if term in text:
                parse = ann_corenlp.sentence[sIndex].binarizedParseTree
                flag= False
                datum=''
                sentence = 'S' + str(len(self.keyPhrases) + 1)
                self.keyPhrases[sentence] = [0]
                chunk_vector = self.get_chunks(parse)
                ibo = 'O'
                key_ibo = 'O'
                for i in range(len(tokens)):
                    token = tokens[i]
                    tag = 'O'
                    keyTerm = False
                    if token.lemma.lower() == term or token.word.lower() == term: keyTerm= True
                    if mode=='cn':
                        span1, span2, rel= spans
                        ibo, phrase_index = self.overlap([span2], (token.beginChar, token.endChar), ibo,
                                                     sentence)
                        key_ibo, _ = self.overlap([span1], (token.beginChar, token.endChar), key_ibo,
                                              sentence)
                        if ibo != 'O': tag = ibo + rel
                        if key_ibo == 'O': keyTerm= False
                    if keyTerm:
                        flag= True
                        args = [str(sIndex + 1), token.word, token.pos, chunk_vector[i], '1', tag]
                    else:
                        args = [str(sIndex + 1), token.word, token.pos, chunk_vector[i], '0', tag]
                    datum += str.join(' ', args) +'\n'
                if flag: data += datum+ '\n'
        return data

    #Not currently used, back-up to save all CN relations
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

    # TODO: Fix that guy to make sense, prettify
    def augment_cn_data(self, data_file):
        f=open(data_file)
        new_data= ast.literal_eval(f.read())
        f.close()
        data=""
        if os.path.exists('../data/ConceptNet/' + 'conceptNet.ibo'):
            f = open('../data/ConceptNet/' + 'conceptNet.ibo')
            data=f.read()
            f.close()
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
                    text= text[0].upper() + text[1:]
                    processed_sentence= self.process_sentence(entity1, text, 'cn', [span1, span2, rel])
                    # ann_corenlp = self.CoreNLPclient.annotate(text[0].upper() + text[1:])
                    # tokens = ann_corenlp.sentence[0].token
                    # deps = ann_corenlp.sentence[0].enhancedPlusPlusDependencies
                    # dep_dic = {deps.root[0] - 1: {'source': -1, 'dep': 'root'}}
                    # for edge in deps.edge:
                    #     dep_dic[edge.target - 1] = {'source': edge.source - 1, 'dep': edge.dep}
                    #
                    # ibo = 'O'
                    # key_ibo='O'
                    #
                    # sentence = 'S' + str(len(self.keyPhrases) + 1)
                    # self.keyPhrases[sentence] = [0]
                    # parse= ann_corenlp.sentence[0].binarizedParseTree
                    # chunkVector= self.get_chunks(parse)
                    #
                    # for i in range(len(tokens)):
                    #     token=tokens[i]
                    #     index = token.tokenBeginIndex
                    #     ##Also changed the ibo to use only span 2 as keyword
                    #     #SO prediction is actually over the span2 and relation (given span1???
                    #     #One more column is thus necessary....???Or not???
                    #     ibo, phrase_index = self.overlap([span2], (token.beginChar, token.endChar), ibo,
                    #                                      sentence)
                    #     key_ibo, _ = self.overlap([span1], (token.beginChar, token.endChar), key_ibo,
                    #                                      sentence)
                    #     tag='O'
                    #     if ibo!='O': tag= ibo+rel
                    #     if key_ibo!= 'O':
                    #         args = [str(i + 1), token.word, token.pos, chunkVector[i], '1', tag]
                    #     else:
                    #         args = [str(i + 1), token.word, token.pos, chunkVector[i], '0', tag]
                    #     datum = str.join(' ', args)
                    #     data += datum + '\n'

                    data += processed_sentence+ '\n'
                except:
                    print('Error')
                    print(item)
        f = open('../data/ConceptNet/' + 'conceptNet.ibo', 'w')
        f.write(data)
        f.close()
        return data

    def split_data(self, data):
        data=data.split('\n\n')[:-1]
        random.shuffle(data)
        size= len(data)
        split= {'train': data[:size*8//10], 'dev':data[size*8//10:size*9//10], 'test':data[size*9//10:]}
        for i in split:
            f=open('../data/ConceptNet/'+i +'.ibo', 'w')
            for datum in split[i]:
                f.write(datum+'\n\n')
            f.close()


    def process_data(self, output_file, term_definition_file=None, definitions=None):
        if definitions==None:
            if term_definition_file== None:
                print("Error: please add file or dictionary...")
                return
            definitions= json.load(open(term_definition_file))
        data = ""
        unprocessed=[]
        counter=0
        for term in definitions:
            counter+=1
            if counter%10==0:
                print('Processing: '+ str(counter)+ '/'+str(len(definitions)))
            text = definitions[term]
            try:
                data+= self.process_sentence(term, text, 'test')
            except:
                print("Could not process term")
                print(term)
                unprocessed.append(term)
        f = open(output_file+'.ibo', 'w')
        f.write(data)
        f.close()

def main():
    parser = argparse.ArgumentParser(description='Augment Data with Wikipedia Links')
    parser.add_argument('--corenlp', type=str, default='/Users/evangeliaspiliopoulou/Desktop/stanfordCoreNLP')
    parser.add_argument('--term_definition_file', type=str, default='../data/term_definitions_wn.json')
    parser.add_argument('--cn_data_file', type=str)
    parser.add_argument('--output_file', type=str, default='../data/FrameTerms_wn')
    parser.add_argument('--mode', type=str, default='test')
    args = parser.parse_args()

    mode= args.mode
    corenlpPath = args.corenlp
    term_definition_file = args.term_definition_file
    output_file= args.output_file

    print('Running Sentence Processing...')
    data_processor= Data_Processor(corenlpPath)
    if mode== 'cn':
        data_file= args.cn_data_file
        data_processor.augment_cn_data(data_file)
    else:
        data_processor.process_data(output_file, term_definition_file=term_definition_file)
    # elif mode== 'ConceptNet':
    #     print('Running CN')
    #     cn_constructor= ConceptNetData(corenlpPath)
    #     #cn_constructor.construct_cn_relations('../data/ConceptNet/allConceptNet.txt')
    #     data= cn_constructor.augment_cn_data('../data/ConceptNet/allConceptNet.txt')
    #     cn_constructor.split_data(data)


if __name__ == '__main__':
    main()
