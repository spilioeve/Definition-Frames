import wikipedia
from PyDictionary import PyDictionary
from nltk.corpus import words
import corenlp
import json
import os
import requests
import argparse




class Definition_Finder:

    def __init__(self, corenlp_path, previous_def_file):
        self.nounTags=['NN', 'NNP', 'NNS', "NNPS",'VBG']
        #os.environ['CORENLP_HOME'] = '/Users/evangeliaspiliopoulou/Desktop/stanfordCoreNLP'
        os.environ['CORENLP_HOME'] = corenlp_path
        self.CoreNLPclient = corenlp.CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma'])
        if previous_def_file!= None:
            self.previous_def = json.load(open(previous_def_file))
        self.exceptionTerms= set()

    def text_expansion_wiki(self, text):
        annotated_sentence = self.CoreNLPclient.annotate(text[0].upper() + text[1:])
        term_list= set()
        for index in range(len(annotated_sentence.sentence)):
            tokens= annotated_sentence.sentence[index].token
            for token in tokens:
                if token.pos in self.nounTags:
                    if token.pos== 'VBG':
                        term= token.value.lower()
                    else:
                        term= token.lemma.lower()
                    term_list.add(term)
        definitions_wiki= self.get_wiki_def(term_list)
        return definitions_wiki

    def get_wiki_def(self, term_list, numSentences=1):
        print('Processing terms in '+str(len(term_list)))
        definitions_wiki = {}
        counter=0
        for term in term_list:
            counter+=1
            if counter%50==0:
                print(counter)
            if term not in self.previous_def:
                r = requests.get('https://en.wikipedia.org/wiki/' + term)
                if r.status_code == 200 and (term not in self.exceptionTerms):
                    try:
                        sentences = wikipedia.summary(term, sentences=numSentences)
                        self.previous_def[term] = sentences
                        definitions_wiki[term] = sentences
                    except:
                        self.exceptionTerms.add(term)
            else:
                definitions_wiki[term]= self.previous_def[term]
        f = open('../data/terms_to_defs.txt', 'w')
        f.write(str(self.previous_def))
        f.close()
        return definitions_wiki

    def get_wn_def(self, term_list):
        english_words = set(words.words())
        print('Processing terms in ' + str(len(term_list)))
        definitions_wn = {}
        counter=0
        dictionary = PyDictionary()
        for term in term_list:
            counter+=1
            if counter%50==0: print(counter)
            #if term in english_words:
            try:
                definition= dictionary.meaning(term)
                if 'Noun' in definition:
                    if len(definition['Noun']) > 0:
                        definition_text= definition['Noun'][0]
                        definitions_wn[term] = term[0].upper() + term[1:] +' is ' +definition_text
                    else:
                        print("Not found: "+ term)
            except:
                print("Not valid english: "+ term)
        return definitions_wn

    def save_definitions(self, def_source, term_list, save_file):
        if os.path.exists(save_file+ '_'+ def_source+'.json'):
            prev_definitions= json.load(open(save_file+ '_'+ def_source+'.json'))
            term_list= term_list-prev_definitions.keys()
        if def_source=='wn':
            definitions= self.get_wn_def(term_list)
        else:
            definitions= self.get_wiki_def(term_list)
        if os.path.exists(save_file+ '_'+ def_source+'.json'):
            prev_definitions.update(definitions)
            definitions= prev_definitions
        json.dump(definitions, open(save_file+ '_'+ def_source+'.json', 'w'))



def main():
    parser = argparse.ArgumentParser(description='Augment Data with Wikipedia Links')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--term_file', type=str, default= '../data/word-sim/ws_terms_nominals')
    parser.add_argument('--save_file', type=str, default='../data/term_definitions')
    parser.add_argument('--def_source', type=str, default='wn')
    parser.add_argument('--previous_def_file', type=str, default=None)
    parser.add_argument('--corenlp', type=str, default='/Users/evangeliaspiliopoulou/Desktop/stanfordCoreNLP')
    args = parser.parse_args()
    corenlpPath= args.corenlp
    def_source= args.def_source
    save_file= args.save_file
    previous_def_file= args.previous_def_file
    dataset= args.dataset
    if dataset== None:
        term_file=args.term_file
        f=open(term_file)
        term_list= f.read().split('\n')[:-1]
        f.close()
        definition_finder = Definition_Finder(corenlpPath, previous_def_file)
        definition_finder.save_definitions(def_source, set(term_list), save_file)

    else:
        print('No dataset instructions given....')




if __name__ == '__main__':
    main()










