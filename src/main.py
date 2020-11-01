from src.wikipedia_links import Definition_Finder
from src.construct_data_RE import Data_Processor
from src.relation_extraction import RelationExtraction
import argparse


#TODO: Fix the methods here to be used from the main, in a cohesive, comprehensive way.....
class DefinitionFrames:
    def __init__(self, corenlpPath, numSentences):
        self.wikiLink= Definition_Finder(corenlpPath)
        self.dataConstructor= Data_Processor(corenlpPath)
        self.numSentences= numSentences
        self.re_model= RelationExtraction(modelType, embedding_type,)


    def extract_frame(self, termList, pretrainedModel= None):
        self.wikiLink.get_wiki_def(termList, self.numSentences)
        wiki_defs={}
        data=''
        for term in termList:
            if term in self.wikiLink.wiki_dic:
                definition= self.wikiLink.wiki_dic[term]
                wiki_defs[term]= definition
                data+= self.dataConstructor(term, definition)
        f=open('../data/testing.ibo')
        f.write(data)
        f.close()
        if pretrainedModel==None:
            print('please add model params')

        else:
            self.re_model.evaluateModel(data)
            ###Just run the RE model here....
            ##Use one of the preloaded models??


    def encodeFrame(self, termList):
        frames = self.readFrames(termList)


def main():
    parser = argparse.ArgumentParser(description='Getting Definition Frame for term(s)')
    parser.add_argument('--corenlp', type=str, default='/Users/evangeliaspiliopoulou/Desktop/stanfordCoreNLP')
    parser.add_argument('--terms', type=list, default=[])
    parser.add_argument('--termFile', type=str)
    parser.add_argument('--encode', type=int, default=1)
    args = parser.parse_args()
    corenlpPath= args.corenlp
    termList = args.terms
    termFile = args.termFile
    if len(args.terms)==0:
        f=open('../data/'+termFile)
        termList=f.read().split('\n')[:-1]
        f.close()
    encode= args.encode
    print('Extracting Wikipedia Data...')
    wikiLink = Wiki_Link_Extractor(corenlpPath)
    def_sentences= wikiLink.getWikiFromTerms(termList, numSentences=2)
    print('Constructing Definitional Data...')
    dataConstructor = WikiData(corenlpPath)
    dataConstructor.constructWikiData('generalTerms', wiki_dict=def_senten/ces)

    print('Constructing Frames...')
    rel_extractor = RelationExtraction('../data/glove.6B.100d.txt', modelType='LSTM', embedding_Dim=200, char_Dim=30, hidden=100,
                                       num_layers=1)




if __name__ == '__main__':
    main()