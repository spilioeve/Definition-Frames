import wikipedia
import corenlp
import json
import os
import requests
import argparse
import ast



class Wiki_Link_Extractor:

    def __init__(self, CoreNLPPath, dataFiles=None):
        self.dataFiles= dataFiles
        self.nounTags=['NN', 'NNP', 'NNS', "NNPS",'VBG']
        #os.environ['CORENLP_HOME'] = '/Users/evangeliaspiliopoulou/Desktop/stanfordCoreNLP'
        os.environ['CORENLP_HOME'] = CoreNLPPath
        self.CoreNLPclient = corenlp.CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma'])
        f = open('../data/terms_to_defs.txt')
        self.wiki_dic = ast.literal_eval(f.read())
        f.close()
        self.exceptionTerms=[]
        #self.questions_to_wikiterms={}


    def getWikiFromText(self, text):
        annotated_sentence = self.CoreNLPclient.annotate(text[0].upper() + text[1:])
        wiki_terms= []
        for index in range(len(annotated_sentence.sentence)):
            tokens= annotated_sentence.sentence[index].token
            for token in tokens:
                if token.pos in self.nounTags:
                    if token.pos== 'VBG':
                        term= token.value.lower()
                    else:
                        term= token.lemma.lower()
                    r = requests.get('https://en.wikipedia.org/wiki/'+term)
                    if r.status_code == 200 and (term not in self.exceptionTerms):
                        if term in self.wiki_dic:
                            wiki_terms.append(term)
                        else:
                            try:
                                sentences = wikipedia.summary(term, sentences=4)
                                self.wiki_dic[term]=sentences
                                wiki_terms.append(term)
                            except:
                                self.exceptionTerms.append(term)
        return wiki_terms

    def getWikiFromTerms(self, termList, numSentences=1):
        print('Processing terms in '+str(len(termList)))
        wiki_sents = {}
        counter=0
        for term in termList:
            counter+=1
            if counter%50==0:
                print(counter)
            if term not in self.wiki_dic:
                r = requests.get('https://en.wikipedia.org/wiki/' + term)
                if r.status_code == 200 and (term not in self.exceptionTerms):
                    try:
                        sentences = wikipedia.summary(term, sentences=numSentences)
                        self.wiki_dic[term] = sentences
                        wiki_sents[term] = sentences
                    except:
                        self.exceptionTerms.append(term)
            else:
                wiki_sents[term]= self.wiki_dic[term]
        f = open('../data/terms_to_defs.txt', 'w')
        f.write(str(self.wiki_dic))
        f.close()
        return wiki_sents

    def getDataQA(self, dataset, file):
        f=open('data/'+file)
        questions= f.read().split('\n')[:-1]
        f.close()
        f = open('data/Wikipedia/'+dataset+'/'+file, 'w')
        current=0
        for q in questions:
            current+=1
            question_json = json.loads(q)
            try:
                print("processing question n" + str(current))
                sentence= question_json['question']['stem']
                wiki_sentence= self.getWikiTerms(sentence)
                question_json['question']['wikiTerms'] = wiki_sentence
                answers = question_json['question']['choices']
                for i in range(len(answers)):
                    wiki_answer= self.getWikiTerms(answers[i]['text'])
                    question_json['question']['choices'][i]['wikiTerms'] = wiki_answer
                out_question= json.dumps(question_json)
                f.write(out_question+'\n')
            except:
                out_question = json.dumps(question_json)
                f.write(out_question + '\n')
                print("Question did not process")
                print(q)
        f.close()


    def getTermsData(self, outputF):
        f=open('../data/terms_to_defs.txt')
        self.wiki_dic= ast.literal_eval(f.read())
        f.close()
        for file in self.dataFiles:
            f=open('../data/'+file)
            lines=f.read().split('\n')[:-1]
            f.close()
        f = open('../data/' + outputF + '.txt', 'w')
        f.write(str(self.wiki_dic))
        f.close()


def main():
    parser = argparse.ArgumentParser(description='Augment Data with Wikipedia Links')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--corenlp', type=str, default='/Users/evangeliaspiliopoulou/Desktop/stanfordCoreNLP')
    args = parser.parse_args()
    corenlpPath= args.corenlp
    dataFiles= args.dataset
    #
    # if dataset=='OpenBook':
    #     data= {'OpenBook':['train.jsonl', 'test.jsonl', 'dev.jsonl']}
    # elif dataset=='ARC-Easy':
    #     data = {'ARC':['ARC-Easy-Train.jsonl', 'ARC-Easy-Dev.jsonl', 'ARC-Easy-Test.jsonl']}
    # elif dataset== 'ARC-Challenge':
    #     data= {'ARC':['ARC-Challenge-Dev.jsonl', 'ARC-Challenge-Train.jsonl', 'ARC-Challenge-Test.jsonl']}
    # else:
    #     print('Dataset is not valid')
    #     print('Please choose options:   ARC-Easy, ARC-Challenge or OpenBook')
    #     return []
    wikiLoader = Wiki_Link_Extractor(dataFiles, corenlpPath)
    print('Processing starts...')
    wikiLoader.writeAllData('terms_to_defs')



if __name__ == '__main__':
    main()










