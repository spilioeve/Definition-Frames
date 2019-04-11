import string
import json
import os
from allennlp.predictors.predictor import Predictor
import pdb
import argparse
from nltk.stem import WordNetLemmatizer
import ast
import random
import csv

class PostProcessData:
    def __init__(self, dataset, getAllen):
        #self.rels = {'IsA': 'is a', 'HasProperty': 'has the property', 'UsedFor': 'is used for', 'PartOf': 'is part of', 'Causes': 'causes', 'DefinedAs':'is defined as', 'MadeOf': 'is made by', 'InstanceOf': 'is an instance of'}
        self.getAllen= getAllen
        self.dataset= dataset
        self.allen_predictor=  Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
        self.lmtzr= WordNetLemmatizer()

    def outputToRelations(self, outputF):
        punct= string.punctuation
        f= open(outputF)
        output= f.read().split('\n\n')[:-1]
        f.close()
        keyTerms={}
        allen_relations={}
        currIndex=0
        sentenceDic={}
        parsed_sentences=[]
        print(len(output))
        print("processing sentences...")
        for sentence in output:
            print(currIndex)
            currIndex+=1
            key=""
            text=""
            relations = {}
            lines= sentence.split('\n')
            sentNumber=1
            for item in lines:
                terms= item.split(' ')
                sentNumber=terms[0]
                word= terms[1]
                text += word + ' '
                flag= terms[2]
                relation = terms[4]
                if flag=='1' and len(key)==0: key = self.lmtzr.lemmatize(word.lower())
                # word= terms[0].lower()
                # text+= word+ ' '
                # relation = terms[2]
                if relation != 'O':
                    start, relation= relation.split('-')
                    if relation not in relations:
                        if word not in punct:
                            relations[relation]= [word]
                    elif start == 'I':
                        prev= relations[relation][-1]
                        relations[relation][-1] = prev +' '+ word
                    else:
                        relations[relation].append(word)
            def_text=key[0].upper()+key[1:]+' '
            if key not in keyTerms:
                keyTerms[key] = []
                sentenceDic[key] =[]
            for r in relations:
                def_text+= r+ ' '
                for term in relations[r]:
                    def_text+= term+ ' AND '
                    keyTerms[key].append((r, term, sentNumber))
            def_text= def_text.strip(' AND ')
            text=text.strip(' ')
            if self.getAllen and text not in parsed_sentences:
                allen_output, allen_text= self.processTextAllen(key, text, sentNumber)
                if key in allen_relations:
                    allen_relations[key]+= allen_output
                else:
                    allen_relations[key] = allen_output
                #def_text = self.expandToSentence(keyTerms[key], sentNumber, key)
                sentenceDic[key].append({'sentence':text, 'allen_text':allen_text, 'def_text':def_text, 'sentenceNumber': sentNumber})
                parsed_sentences.append(text)
        return keyTerms, allen_relations, sentenceDic

    def mergeRelationsToQA(self, files):
        wiki_terms, allen_relations, sentenceDic = self.outputToRelations('data/output/wiki_'+self.dataset+'.ibo')
        f=open('data/'+self.dataset+'/wiki_def_relations.txt', 'w')
        f.write(str(wiki_terms))
        f.close()
        if self.getAllen:
            f= open('data/'+self.dataset+'/wiki_allen_relations.txt', 'w')
            f.write(str(allen_relations))
            f.close()
        f = open('data/' + self.dataset + '/sentence_encodings.txt', 'w')
        f.write(str(sentenceDic))
        f.close()
        for file in files:
            print(file)
            f = open('data/' + 'Wikipedia/' + self.dataset +'/'+ file)
            questions = f.read().split('\n')[:-1]
            f.close()
            try:
                os.stat('data/' + self.dataset + '/new')
            except:
                os.mkdir('data/' + self.dataset + '/new')
            f = open('data/' + self.dataset + '/new/' + file, 'w')
            for i in range(len(questions)):
                question = questions[i]
                question_json = json.loads(question)
                print("processing question n" + str(i) + str(question_json['id']))
                terms_question = question_json['question']['wikiTerms']
                question_dic = {}
                for term in terms_question:
                    term = self.lmtzr.lemmatize(term.lower())
                    if term in sentenceDic:
                        descriptions = sentenceDic[term]
                        question_dic[term]= descriptions
                        #term_rels = []
                        # for description in descriptions:
                        #     term_rels.append({'definition': description[0], 'relation': description[1],
                        #                       'sentNumber': int(description[2])})
                        #question_dic[term] = term_rels
                question_json['question']['wikiTerms'] = question_dic
                answers = question_json['question']['choices']
                for j in range(len(answers)):
                    answer_dic = {}
                    terms_answer = question_json['question']['choices'][j]['wikiTerms']
                    for term in terms_answer:
                        term= self.lmtzr.lemmatize(term.lower())
                        if term in sentenceDic:
                            descriptions = sentenceDic[term]
                            # term_rels = []
                            # for description in descriptions:
                            #     term_rels.append({'definition': description[0], 'relation': description[1],
                            #                       'sentNumber': int(description[2])})
                            # answer_dic[term] = term_rels
                            answer_dic[term]= descriptions
                    question_json['question']['choices'][j]['wikiTerms'] = answer_dic
                question_output = json.dumps(question_json)
                f.write(question_output + '\n')
            f.close()

    def processTextAllen(self, keyword, text, sentNumber):
        allen_text= keyword[0].upper() + keyword[1:]+' '
        events=[]
        #allen_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")
        allen_output = self.allen_predictor.predict(text)
        for index in range(len(allen_output['verbs'])):
            try:
                verb= allen_output['verbs'][index]['verb']
                description= allen_output['verbs'][index]['description']
                if 'ARG0:' in description and 'ARG1' in description:
                    i = description.index('ARG0: ')
                    subStr0 = description[i:]
                    ii = subStr0.index(']')
                    arg0 = description[i + 6:ii+i]
                    j = description.index('ARG1: ')
                    subStr1 = description[j:]
                    jj = subStr1.index(']')
                    arg1 = description[j + 6:jj + j]
                    #arg0, arg1 = self.findArgs(allen_output['verbs'][i]['description'])
                    ##This line might need fixing... If arg0 is not one word???
                    if keyword.lower() in arg0.lower():
                        events.append((verb, arg1, sentNumber))
                        allen_text+= verb+ ' '+ arg1 + ' AND '
            except:
                pdb.set_trace()
        allen_text= allen_text.strip(' AND ')
        return events, allen_text

def translateAMT(inputF, num_samples, size):
    f=open('data/Amt_study/'+inputF)
    csv_reader = csv.DictReader(f)
    output={}
    possible_answers = ["1 - System 1", "2 - System 2", "3 - Cannot tell: Both are equally good/bad"]
    for row in csv_reader:
        sentence= row["Input.Sentence"]
        reverse= row["Input.Reverse"]
        sentNum= row["Input.SentenceNumber"]
        answer= row["Answer.Compare Quality of System Output.label"]
        #answer= possible_answers.index(answer)
        if sentence not in output:
            output[sentence]={i:0 for i in possible_answers}
            output[sentence].update({"sentNum": int(sentNum), "reverse": int(reverse)})
        output[sentence][answer] += 1
    f.close()
    details=[]
    results={i:0 for i in range(len(possible_answers))}
    #annotator_argr= {i:{0:0, 2:0, 3:0} for i in range(len(possible_answers))}
    sentenceBatch = list(output.keys())
    #fleiss_vector = []
    #sentences = list(output.keys())
    #for j in range(num_samples):
    #random.shuffle(sentences)
    #sentenceBatch= sentences[:size]
    category_scores={i:0 for i in range(len(possible_answers))}
    datum_scores={sentence:0 for sentence in sentenceBatch}
    for sentence in sentenceBatch:
        f_answer=1 ##Default is 2
        score=0
        for i in possible_answers:
            count= output[sentence][i]
            answer= possible_answers.index(i)
            if output[sentence]["reverse"]:
                if answer == 1: answer=0
                elif answer==0: answer=1
            category_scores[answer]+= count
            datum_scores[sentence]+=count*(count-1)
            if count>1:
                f_answer= answer
                score=count
                # if output[sentence][possible_answers[-1]]==1:
                #     score+=1
        datum_scores[sentence] /= (len(possible_answers)*(len(possible_answers)-1))
        details.append((f_answer, score, output[sentence]["sentNum"]))
        #annotator_argr[f_answer][score]+= 1
        results[f_answer]+=1
    category_mean= 0
    for i in category_scores:
        category_mean+= (category_scores[i]/(len(possible_answers)*len(sentenceBatch)))**2
    datum_mean=0
    for sentence in sentenceBatch:
        datum_mean+= datum_scores[sentence]
    datum_mean /= len(sentenceBatch)
    fleiss= (datum_mean - category_mean) / (1-category_mean)
    # fleiss_vector.append(fleiss)
    # fleiss_vector.append(0.1)
    # zsc = st.zscore(fleiss_vector)
    # return zsc[-1]
    # pdb.set_trace()
    return details, results, fleiss







def AMTExperiment(numberExamples=200):
    rels = {'IsA': 'is a', 'HasProperty': 'has the property', 'UsedFor': 'is used for', 'PartOf': 'is part of',
            'Causes': 'causes', 'DefinedAs': 'is defined as', 'MadeOf': 'is made by', 'InstanceOf': 'is an instance of'}
    f = open('data/OpenBook/sentence_encodings.txt')
    textualDic = ast.literal_eval(f.read())
    f.close()
    f = open('data/OpenBook/wiki_def_relations.txt')
    defDic = ast.literal_eval(f.read())
    f.close()
    f = open('data/OpenBook/wiki_allen_relations.txt')
    allenDic = ast.literal_eval(f.read())
    f.close()
    emptyAllen=0
    empty_RE=0
    terms=list(textualDic.keys())
    for key in terms:
        def_i = defDic[key]
        text_i = textualDic[key]
        allen_i = allenDic[key]
        for i in text_i:
            sentNum= i['sentenceNumber']
            allen2 = [j for j in allen_i if j[2] == sentNum]
            def2 = [j for j in def_i if j[2] == sentNum]
            if len(allen2)==0:
                emptyAllen+=1
            if len(def2)==0:
                empty_RE+=1
    print("empty Allen: "+ str(emptyAllen))
    print('empty RE '+ str(empty_RE))
    ##Statistics:
    #empty Allen: 1438
    #empty RE 11
    random.shuffle(terms)
    terms= terms[:numberExamples]
    terms2= terms[:]
    random.shuffle(terms2)
    flagVector= [random.randint(0,1) for i in range(2*numberExamples)]
    dataPerSentence={'1':0, '2':0, '3':0, '4':0, '5':0, '6':0}
    f = open('data/AMT_study/input_data.csv', 'w')
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['KeyTerm', 'Sentence', 'System1', 'System2', 'Reverse', 'SentenceNumber'])
    for k in range(numberExamples):
        key = terms[k]
        def_i = defDic[key]
        allen_i = allenDic[key]
        text_i = textualDic[key]
        sentenceNum = text_i[0]['sentenceNumber']
        def1 = ['Relation: ' + rels[i[0]] + ', Related to: ' + i[1] for i in def_i if i[2] == sentenceNum]
        allen1 = ['Relation: ' + i[0] + ', Related to: ' + i[1] for i in allen_i if i[2] == sentenceNum]
        sent1 = toASCII(text_i[0]['sentence'])
        flag = flagVector[2*k]
        flag2 = flagVector[2 * k + 1]
        if len(allen1)>0 and len(def1)>0:
            dataPerSentence[sentenceNum]+=1
            if flag:
                writer.writerow([key, sent1, toASCII(str.join('<br/>', def1)), toASCII(str.join('<br/>', allen1)), 0, int(sentenceNum)])
            else:
                writer.writerow([key, sent1, toASCII(str.join('<br/>', allen1)), toASCII(str.join('<br/>', def1)), 1, int(sentenceNum)])
        key2 = terms2[k]
        text_i = textualDic[key2]
        if len(text_i) > 2:
            def_i = defDic[key2]
            allen_i = allenDic[key2]
            sent2 = toASCII(text_i[1]['sentence'])
            sentenceNum2= text_i[1]['sentenceNumber']
            def2 = ['Relation: ' + rels[i[0]] + ', Related to: ' + i[1] for i in def_i if i[2] == sentenceNum2]
            allen2 = ['Relation: ' + i[0] + ', Related to: ' + i[1] for i in allen_i if i[2] == sentenceNum2]
            if len(allen2) > 0 and len(def2) > 0:
                dataPerSentence[sentenceNum2] += 1
                if flag2:
                    writer.writerow([key2, sent2, toASCII(str.join('<br/>', def2)), toASCII(str.join('<br/>', allen2)), 0, int(sentenceNum2)])
                else:
                    writer.writerow([key2, sent2, toASCII(str.join('<br/>', allen2)), toASCII(str.join('<br/>', def2)), 1, int(sentenceNum2)])
    f.close()
    print('Number of data per Sentence Index:  '+ str(dataPerSentence))


def toASCII(text):
    newText=""
    for t in text:
        if ord(t) <127:
            newText+= t
    newText= newText.encode('utf-8').decode('utf-8')
    return newText


def main():
    parser = argparse.ArgumentParser(description='Post-Processing Module')
    parser.add_argument('--dataset', type=str, default='OpenBook', help='Dataset')
    parser.add_argument('--path', type=str, default='', help='Path')
    parser.add_argument('--allenIE', type= bool, default= True)

    args = parser.parse_args()
    dataset=args.dataset
    allenIE= args.allenIE

    dataprocess = PostProcessData(dataset, allenIE)
    #files= os.listdir('data/'+dataset)
    if dataset=='OpenBook':
        files= ['train.jsonl', 'test.jsonl', 'dev.jsonl']
    else:
        files= ['ARC-Challenge-Dev.jsonl', 'ARC-Easy-Train.jsonl', 'ARC-Easy-Dev.jsonl', 'ARC-Challenge-Train.jsonl',
         'ARC-Easy-Test.jsonl', 'ARC-Challenge-Test.jsonl']
    #files = [i for i in files if i != '.DS_Store']
    dataprocess.mergeRelationsToQA(files)


#keywords= dataprocess.outputToRelations('data/Wikipedia/'+dataset+'.ibo', 'data/output/wiki.ibo', 'data/'+dataset+'/wiki_relations.txt', batch_size)

if __name__ == '__main__':
    main()

def processMeta(file):
    f=open(file)
    lines=f.read().split('\n')[:-1]
    f.close()
    f=open(file, 'w')
    for line in lines:
        question= json.loads(line)
        terms_question= question['question']['wikiTerms']
        empty=True
        for term in terms_question:
            items= terms_question[term]
            if len(items)>0: empty=False
            newItems=[]
            for description in items:
                if len(description['def_text'])==0:
                    description['def_text']= term
                if len(description['allen_text'])==0:
                    description['allen_text']= term
                newItems.append(description)
            question['question']['wikiTerms'][term]=newItems
        if len(terms_question) == 0 or empty:
            question['question']['wikiTerms']={'.':[
                {'def_text': '.', 'allen_text': '.', 'sentence': '.', 'sentenceNumber': "1"}]}
        answers= question['question']['choices']
        for j in range(len(answers)):
            terms_answer = question['question']['choices'][j]['wikiTerms']
            #text= question['question']['choices'][j]['text']
            empty = True
            for term in terms_answer:
                items = terms_answer[term]
                if len(items) > 0: empty = False
                newItems = []
                for description in items:
                    if len(description['def_text']) == 0:
                        description['def_text'] = term
                    if len(description['allen_text']) == 0:
                        description['allen_text'] = term
                    newItems.append(description)
                question['question']['choices'][j]['wikiTerms'][term] = newItems
            if len(terms_answer)==0 or empty:
                question['question']['choices'][j]['wikiTerms']={'.': [
                        {'def_text': '.', 'allen_text': '.', 'sentence': '.', 'sentenceNumber': "1"}]}
        question_output= json.dumps(question)
        f.write(question_output+'\n')
    f.close()
