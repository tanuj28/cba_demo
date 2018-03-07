from flask import Flask, request, url_for, redirect, render_template
import numpy as np
import pandas as pd
##from itertools import chain
import pandas as pd
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
from random import sample
import re


#nltk.download()

#from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
import re
import matplotlib.pyplot as plt
from time import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

suffix = '.'

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from nltk import pos_tag
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import pandas as pd

import nltk
#nltk.download('punkt')
from nltk import sent_tokenize


style.use('fivethirtyeight')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
from nltk.chunk import conlltags2tree, tree2conlltags


from nltk.tag import StanfordNERTagger
from keras.models import model_from_json

#from spyre import server

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
import re
import matplotlib.pyplot as plt
import itertools
from time import time
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import StratifiedShuffleSplit
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()

from sklearn.model_selection import StratifiedKFold

from random import shuffle



from sklearn.feature_extraction.text import CountVectorizer as cv, TfidfVectorizer as tfidf





from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder







# In[ ]:


import os  
java_path = "tanuj28/cba_demo"
os.environ['JAVAHOME'] = java_path


# In[ ]:


st = StanfordNERTagger('tanuj28/cba_demo/english.conll.4class.distsim.crf.ser.gz',
                       'tanuj28/cba_demo/stanford-ner.jar',
                           encoding='utf-8')

st1 = StanfordNERTagger('tanuj28/cba_demo/english.conll.4class.distsim.crf.ser.gz',
                       'tanuj28/cba_demo/stanford-ner.jar',
                           encoding='utf-8')

# In[14]:


def prep1_email (words):
    final1_test = []
    sub = ((((((words.replace(",", " ")).replace(" ( ", " ")).replace(" ) ", " ")).replace(" . ", " ")).replace(" .. ", " ")).replace(" - ", " ")).split() 
    sub = list(filter(lambda x : x != '.',sub))
    for j in range (len(sub)):
            sub[j] = sub[j].rstrip('.')
            sub[j] = sub[j].rstrip(',')
            sub[j] = sub[j].rstrip(')')
            sub[j] = sub[j].lstrip('(')
            sub[j] = sub[j].lstrip('"')
            sub[j] = sub[j].rstrip('"')
    final1_test.append(sub)
    return final1_test
def prep2_email (final1_test):
    final2_test = []
    scor1_test = []
    for i in range(len(final1_test)):
            scor1_test = []
            sub_test = pos_tag(final1_test[i])
            iob_tagged_test = tree2conlltags(sub_test)
            for j in range(len(iob_tagged_test)):
                scor1_test.append(iob_tagged_test[j])

            final2_test.append(scor1_test)
            del scor1_test
    return final2_test


def prep1_reci (words):
        final1_test = []
    
        p = words
       #p = p.replace(".", " ")
        p = p.replace(',','')
        p = p.replace(':','')
        p = p.replace('RM-','')
        p = p.replace('"','')
    
        p = p.replace('-','')
        p = p.replace('(','')
        p = p.replace(')','')
        p = p.replace('/','')
        p = p.replace('*','')
        p = p.replace('RM','')
        p = p.replace('BM','')
        p = p.replace('$','')
        p = p.replace('&','')
        p = p.replace(';','')
        #p = p.replace('directly', '')
        p = p.replace('..','')
        #p = p.replace('analyst','')
        #p = p.replace('asap','')
        p = p.replace('BU', '')
        p = p.replace('BSB', 'bsb')
        p = p.replace('Branch', 'branch')
        p = p.replace('Manager', 'manager')
        p = p.replace('Banker', 'banker')
        p = p.replace('cc', 'and to')
        p = p.replace('email', 'email to')
        p = p.replace('letter', 'documents')
        p = p.replace('docs', 'documents')
        p = p.replace('ASAP', 'asap')
        p = p.replace("Search", "search")
        p = p.replace("Road", "road")
        p = p.replace("Rd", "rd")
        p = p.replace(' me ',' ')
        p = p.replace('Thanks. Regards', 'thanks')
        p = p.replace('Thank you', 'thanks')
        p = p.replace(' Cheers ', ' thanks ')
        p = p.replace(' uit ', ' unit ')
        p = p.replace(' it ', '')
        p = p.replace('docunts', 'documents')
    
    
        p = p.split()
        #print(p, len(p))
        for j in range(len(p)):
            q = p[j]
            if (str("@") in q and len(q) != 1):
                #print("@present", q)
                r = q.split(".")
                if (( len(r) == 4) or (len(r)==3 and r[1] != "com")):
                    r[0] = r[0].title()
                    r[0] = r[0].lstrip('"')
                    r[0] = r[0].rstrip('"')
                    r[1] = r[1].split("@")
                    r[1][0] = r[1][0].title()
                    #print(r[1][0])
                    x = p.index(q) # This raises ValueError if there's no 'b' in the list.
                    p[j:j+1] = r[0], r[1][0]                
                elif(len(r) == 2):
                    r[0] = r[0].split("@")
                    r[0][0] = r[0][0].title()
                    p[j] = r[0][0]
                
            elif ((len(q)!= 1 and q[len(q) - 1] == ".") or (len(q)!= 1 and q[len(q) - 1] == "?")):
                #print("occr")
                p[j] = p[j].rstrip('.')
                p[j] = p[j].rstrip('?')
                if (j < len(p) - 2):
                    if (p[j+2].istitle()==False):
                        p[j+1] = p[j+1].lower()

            if( p[j]=="Thanks" or p[j]=="PO"  or p[j]== "Contact" or p[j]=="Regards" or p[j]==" EMAIL" or p[j]=="Business" or p[j]=="Cheers" or p[j]=="Multiple" or p[j]=="Director" or p[j]=="Thank" or p[j]=="Com" or p[j]=="Kind"  or p[j]=="Internal" or p[j] == "Mailing"):
                p[j] = p[j].lower()
            
            elif((p[j].isupper()) ):
                p[j] = p[j].lower()
            if (p[j]=='and' and p[j-1].istitle()):
                p[j:j+1] = 'and', 'to'
        if (str('@') in p):
            p.remove('@')
        p = [item for item in p if not item.isdigit()]
        p = [item for item in p if not len(item)==1]
        final1_test.append(p)
        return final1_test
    
    
def prep2_reci (final1_test):
    final2_test = []
    scor1_test = []
    for i in range(len(final1_test)):
        scor1_test = []
        sub_test = pos_tag(final1_test[i])
        iob_tagged_test = tree2conlltags(sub_test)
        st_tag_test = st.tag(final1_test[i])
        for j in range(len(iob_tagged_test)):
            a_test = tuple([st_tag_test[j][1]])
            iob_tagged_test[j] = iob_tagged_test[j] + (a_test)
            scor1_test.append(iob_tagged_test[j])
    
        final2_test.append(scor1_test)
        del scor1_test
    return final2_test



def prep1_add(words):
    final1_test = []
    p = words
    p = p.replace('*', '')
    p = p.replace('(', '')
    p = p.replace(')', '')
    p = p.replace(' . ', ' ')
    p = p.replace(' .. ', '')
    p = p.replace('RM', '')
    p = p.replace('Address', 'address to')
    p = p.replace('address', 'address to')
    p = p.replace(':', '')
    p = p.replace(' me ', ' ')
    p = p.replace(' at ', ' to ')
    p = p.replace(' us ', ' ')
    p = p.replace(' - ', ' ')
    p =  p.replace(' @ ', ' ')
    sub = p.split() 
    for j in range (len(sub)):
        sub[j] = sub[j].rstrip('.')
        sub[j] = sub[j].rstrip(')')
        sub[j] = sub[j].lstrip('(')
        if ((sub[j] == 'BSB') | (sub[j] == '(BSB)')):
            sub[j] = 'BSB:'
        if ((str('BSB') in sub[j]) and len(sub[j])>4):
            a = re.split('(\d.*)',sub[j])
            if (a[0]== 'BSB'):
                a[0] = 'BSB:'
            sub[j:j+1] = a[0],a[1]
        if (sub[j]=='BSB:' and sub[j+1] == 'is'):
            sub[j]='address'
            sub[j+1] = 'BSB:'
        if ((sub[j].rstrip(',')).isdigit() and len(sub[j])==4 and sub[j-1]=='BSB:'):
            sub[j] = sub[j].rstrip(',')
        if(j>3):
            if (sub[j].isdigit() and sub[j-1].isdigit() and sub[j-2].isdigit()):
                sub[j] = ' '
                sub[j-1] = ' '
                sub[j-2] = ' '
        if (sub[j]=='CDL' or sub[j]== 'NEW' or sub[j]== 'LOAN' or sub[j]=='DOCS' or sub[j]=='PREPARED' or sub[j]=='IN'):
            sub[j] = sub[j].lower()
    sub = [item for item in sub if  not item == ' ']
    #print(i)
    final1_test.append(sub)
    return final1_test

def prep2_add(final1_test):
    final2_test = []
    scor1_test = []
    for i in range(len(final1_test)):
        scor1_test = []
        sub_test = pos_tag(final1_test[i])
        iob_tagged_test = tree2conlltags(sub_test)
        st_tag_test = st1.tag(final1_test[i])
        for j in range(len(iob_tagged_test)):
            a_test = tuple([st_tag_test[j][1]])
            iob_tagged_test[j] = iob_tagged_test[j] + (a_test)
            scor1_test.append(iob_tagged_test[j])
    
        final2_test.append(scor1_test)
        del scor1_test
    return final2_test


# a bigram function is defined
##import nltk
from nltk.util import ngrams

def grams(word5,word6, min=2, max=3):
    words= []
    words.append(word5)
    words.append(word6)
    #words.append(word2)
    #words.append(word3)
    s = []
    for n in range(min, max):
        for ngram in ngrams(words, n):
            s.append(' '.join(str(i) for i in ngram))
    return s


# In[17]:


# a function is defined for detemining the shape of the token
import re
 
def shape(word):
    word_shape = 'other'
    
    if re.match('\W+$', word):
        word_shape = 'punct'
        
    elif re.match('[A-Z][a-z]+[A-Z][a-z]+[A-Za-z]*$', word):
        word_shape = 'camelcase'
    elif re.match('[A-Za-z]+$', word):
        word_shape = 'mixedcase'
    elif re.match('__.+__$', word):
        word_shape = 'wildcard'
    elif re.match('[A-Za-z0-9]+\.$', word):
        word_shape = 'ending-dot'
    elif re.match('[A-Za-z0-9]+\.[A-Za-z0-9\.]+\.$', word):
        word_shape = 'abbreviation'
    elif re.match('[A-Za-z0-9]+\-[A-Za-z0-9\-]+.*$', word):
        word_shape = 'contains-hyphen'
 
    return word_shape


# In[18]:


# a function to check whether "@" is present in a token or not
def contains_at(word):
    p = False
    if ((str("@") in word) and (len(word) != 1)):
        p = True
    return p


# In[19]:


# a function is defined to check if a word is "to" or not
def prev_word(word):
    p = False
    if (word == "to"):
        p = True
    return p


# In[20]:


# a function defined to check id the nltk ner tag is "PERSON" or not 
def ner_person(word):
    ner_word = False
    
    if word == 'B-PERSON' or word == 'I-PERSON':
        ner_word = True
        
    return ner_word


# In[21]:


# all the features a defined for each token and with a window size of 7
def word2features_email(sent,i):
    word = sent[i][0]
    postag = sent[i][1]
    ner_tag = sent[i][2] 


    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.islower()': word.islower(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isalphanum()': word.replace('-',"").replace('/',"").isalnum(),
        'word.start_captial':word[0].isupper(),
        'word.isalpha()':word.replace('-',"").replace('/',"").isalpha(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word.shape()':shape(word),
        'word.person':ner_person(ner_tag),
        #'word.org':ner_org(ner_tag),
        'word[-3:]': word[-3:],
        'word[-4:]': word[-4:],
        'word[:3]': word[:3],
        'word[:4]': word[:4],
        'word.is_to': prev_word(word),
        'word.cont_at' : contains_at(word)
        
    }
    
    if i > 2:
        word0 = sent[i][0]
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        ner_tag1 = sent[i-1][2]
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        ner_tag2 = sent[i-2][2]
        word3 = sent[i-3][0]
        postag3 = sent[i-3][1]
        ner_tag3 = sent[i-3][2]
       
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.person':ner_person(ner_tag1),
            #'-1:word.org':ner_org(ner_tag1),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.islower()': word1.islower(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:word.isdigit()':word1.isdigit(),  
            '-1:word.isalphanum()': word1.replace('-',"").replace('/',"").isalnum(),
            '-1:word.start_captial':word1[0].isupper(),
            '-1:word.isalpha()':word1.replace('-',"").replace('/',"").isalpha(),  
            '-1:word[-3:]': word1[-3:],
            '-1:word[-4:]': word1[-4:],
            '-1:word[:3]': word1[:3],
            '-1:word[:4]': word1[:4],
            '-1:word.is_to': prev_word(word1),
            '-1:word.shape()':shape(word1),
            '-1:word.grams':grams(word1,word),
            
            
            '-2:word.start_captial':word2[0].isupper(),
            '-2:word.lower()': word2.lower(),
            '-2:word.person':ner_person(ner_tag2),
            #'-2:word.org':ner_org(ner_tag2),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.islower()': word2.islower(),
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],
            '-2:word[-3:]': word2[-3:],
            '-2:word[-4:]': word2[-4:],
            '-2:word[:3]': word2[:3],
            '-2:word[:4]': word2[:4], 
            '-2:word.is_to': prev_word(word2),
            '-2:word.isdigit()':word2.isdigit(),
            '-2:word.isalphanum()':word2.replace('-',"").replace('/',"").isalnum(),
            '-2:word.isalpha()':word2.replace('-',"").replace('/',"").isalpha(),
            '-2:word.grams':grams(word2,word1),
            '-2:word.shape()':shape(word2),
            
            '-3:word.start_captial':word3[0].isupper(),
            '-3:word.person':ner_person(ner_tag3),
            #'-3:word.org':ner_org(ner_tag3),
            '-3:word.lower()': word3.lower(), 
            '-3:word.istitle()': word3.istitle(),
            '-3:word.isupper()': word3.isupper(),
            '-3:word.islower()': word3.islower(),
            '-3:postag': postag3,
            '-3:postag[:2]': postag3[:2],
            '-3:word.isdigit()':word3.isdigit(),
            '-3:word.isalphanum()':word3.replace('-',"").replace('/',"").isalnum(),
            '-3:word.isalpha()':word3.replace('-',"").replace('/',"").isalpha(),
            '-3:word.shape()':shape(word3),
            '-3:word[-3:]': word3[-3:],
            '-3:word[-4:]': word3[-4:],
            '-3:word[:3]': word3[:3],
            '-3:word[:4]': word3[:4],
            '-3:word.is_to': prev_word(word3),
            '-3:word.grams':grams(word3,word2),
            
            
           

        })
        
   
    elif i > 1:
        word0 = sent[i][0]
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        ner_tag1 = sent[i-1][2]
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        ner_tag2 = sent[i-2][2]
        
        features.update({
            '-1:word.lower()': word1.lower(), 
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.islower()': word1.islower(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:word.isdigit()':word1.isdigit(),  
            '-1:word.isalphanum()': word1.replace('-',"").replace('/',"").isalnum(),
            '-1:word.start_captial':word1[0].isupper(),
            '-1:word.isalpha()':word1.replace('-',"").replace('/',"").isalpha(),
            '-1:word.person':ner_person(ner_tag1),
            #'-1:word.org':ner_org(ner_tag1),
            '-1:word.shape()':shape(word1),
            '-1:word[-3:]': word1[-3:],
            '-1:word[-4:]': word1[-4:],
            '-1:word[:3]': word1[:3],
            '-1:word[:4]': word1[:4],
            '-1:word.is_to': prev_word(word1),
            '-1:word.grams':grams(word1,word),
            
            
            
            '-2:word.start_captial':word2[0].isupper(),
            '-2:word.lower()': word2.lower(), 
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.islower()': word2.islower(),
            '-2:word.person':ner_person(ner_tag2),
            #'-2:word.org':ner_org(ner_tag2),
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],
            '-2:word.isdigit()':word2.isdigit(),
            '-2:word.isalphanum()':word2.replace('-',"").replace('/',"").isalnum(),
            '-2:word.isalpha()':word2.replace('-',"").replace('/',"").isalpha(),
            '-2:word.shape()':shape(word2),
            '-2:word[-3:]': word2[-3:],
            '-2:word[-4:]': word2[-4:],
            '-2:word[:3]': word2[:3],
            '-2:word[:4]': word2[:4],
            '-2:word.is_to': prev_word(word2),
            '-2:word.grams':grams(word2,word1),
            
                
        })
    
    elif i > 0:
        word0 = sent[i][0]
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        ner_tag1 = sent[i-1][2]
        
        features.update({
            '-1:word.lower()': word1.lower(), 
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.islower()': word1.islower(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:word.isdigit()':word1.isdigit(),  
            '-1:word.isalphanum()': word1.replace('-',"").replace('/',"").isalnum(),
            '-1:word.start_captial':word1[0].isupper(),
            '-1:word.isalpha()':word1.replace('-',"").replace('/',"").isalpha(),
            '-1:word.shape()':shape(word1),
            '-1:word.person':ner_person(ner_tag1),
            #'-1:word.org':ner_org(ner_tag1),
            '-1:word[-3:]': word1[-3:],
            '-1:word[-4:]': word1[-4:],
            '-1:word[:3]': word1[:3],
            '-1:word[:4]': word1[:4],
            '-1:word.is_to': prev_word(word1),
            '-1:word.grams':grams(word1,word),
            
                
            })
            
       
    
    else:
        features['BOS'] = True

        
        
   
    if i < len(sent) -3:
        word0 = sent[i][0]
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        ner_tag1 = sent[i+1][2]
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        ner_tag2 = sent[i+2][2]
        word3 = sent[i+3][0]
        postag3 = sent[i+3][1]
        ner_tag3 = sent[i+3][2]
    
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.islower()': word1.islower(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:word.isdigit()':word1.isdigit(),
            '+1:word.person':ner_person(ner_tag1),
            #'+1:word.org':ner_org(ner_tag1),
            '+1:word.isalphanum()': word1.replace('-',"").replace('/',"").isalnum(),
            '+1:word.start_captial':word1[0].isupper(),
            '+1:word.isalpha()':word1.replace('-',"").replace('/',"").isalpha(),
            '+1:word.shape()':shape(word1),
            '+1:word[-3:]': word1[-3:],
            '+1:word[-4:]': word1[-4:],
            '+1:word[:3]': word1[:3],
            '+1:word[:4]': word1[:4],
            '+1:word.grams':grams(word,word1),
            
            
            
            '+2:word.start_captial':word2[0].isupper(),
            '+2:word.isdigit()':word2.isdigit(),
            '+2:word.lower()': word2.lower(),
            '+2:word.person':ner_person(ner_tag2),
            #'+2:word.org':ner_org(ner_tag2),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.islower()': word2.islower(),
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],
            '+2:word.isalphanum()': word2.replace('-',"").replace('/',"").isalnum(),
            '+2:word.isalpha()':word2.replace('-',"").replace('/',"").isalpha(),
            '+2:word.shape()':shape(word2),
            '+2:word[-3:]': word2[-3:],
            '+2:word[-4:]': word2[-4:],
            '+2:word[:3]': word2[:3],
            '+2:word[:4]': word2[:4],
            '+2:word.grams':grams(word1,word2),
            
            '+3:word.start_captial':word3[0].isupper(),
            '+3:word.lower()': word3.lower(),
            '+3:word.istitle()': word3.istitle(),
            '+3:word.isupper()': word3.isupper(),
            '+3:word.islower()': word3.islower(),
            '+3:word.person':ner_person(ner_tag3),
            #'+3:word.org':ner_org(ner_tag3),
            '+3:postag': postag3,
            '+3:postag[:2]': postag3[:2],
            '+3:word.isdigit()':word3.isdigit(),
            '+3:word.isalphanum()':word3.replace('-',"").replace('/',"").isalnum(),
            '+3:word.isalpha()':word3.replace('-',"").replace('/',"").isalpha(),
            '+3:word.shape()':shape(word3),
            '+3:word[-3:]': word3[-3:],
            '+3:word[-4:]': word3[-4:],
            '+3:word[:3]': word3[:3],
            '+3:word[:4]': word3[:4],
            '+3:word.grams':grams(word2,word3),
            
                   
            })
    
    
    elif i < len(sent)- 2:
        word0 = sent[i][0]
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        ner_tag1 = sent[i+1][2]
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        ner_tag2 = sent[i+1][2]
        
        
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.islower()': word1.islower(),
            '+1word.person':ner_person(ner_tag1),
            #'+1:word.org':ner_org(ner_tag1),
            '+1:word[-3:]': word1[-3:],
            '+1:word[-4:]': word1[-4:],
            '+1:word[:3]': word1[:3],
            '+1:word[:4]': word1[:4],
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:word.isdigit()':word1.isdigit(),
            '+1:word.isalphanum()': word1.replace('-',"").replace('/',"").isalnum(),
            '+1:word.isalpha()':word1.replace('-',"").replace('/',"").isalpha(),
            '+1:word.shape()':shape(word1),
            '+1:word.start_captial':word1[0].isupper(),
            '+1:word.grams':grams(word,word1),
            
            
            
            '+2:word.start_captial':word2[0].isupper(),
            '+2:word.isdigit()':word2.isdigit(),
            '+2:word.lower()': word2.lower(), 
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.islower()': word2.islower(),
            '+2:word.person':ner_person(ner_tag2),
            #'+2:word.org':ner_org(ner_tag2),
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],
            '+2:word.isalphanum()': word2.replace('-',"").replace('/',"").isalnum(),
            '+2:word.isalpha()':word2.replace('-',"").replace('/',"").isalpha(),
            '+2:word.shape()':shape(word2),
            '+2:word[-3:]': word2[-3:],
            '+2:word[-4:]': word2[-4:],
            '+2:word[:3]': word2[:3],
            '+2:word[:4]': word2[:4],
            '+2:word.grams':grams(word1,word2),
            
                
            
        })        
    
    elif i < len(sent) -1:
        word0 = sent[i][0]
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        ner_tag1 = sent[i+1][2]
    
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.islower()': word1.islower(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:word.isdigit()':word1.isdigit(),
            '+1:word.person':ner_person(ner_tag1),
            #'+1:word.org':ner_org(ner_tag1),
            '+1:word.isalphanum()': word1.replace('-',"").replace('/',"").isalnum(),
            '+1:word.start_captial':word1[0].isupper(),
            '+1:word.isalpha()':word1.replace('-',"").replace('/',"").isalpha(),
            '+1:word.shape()':shape(word1),
            '+1:word[-3:]': word1[-3:],
            '+1:word[-4:]': word1[-4:],
            '+1:word[:3]': word1[:3],
            '+1:word[:4]': word1[:4],
            '+1:word.grams':grams(word,word1),
            
                   
            })
    
    
    else:
        features['EOS'] = True

    return features


# In[22]:


def sent2features_email(sent):
            return [word2features_email(sent,i) for i in range(len(sent))]


# In[ ]:


def prep_email(final1_test):
    final2_test = prep2_email(final1_test)
    return final2_test

def make_fin1_email(mail):
    final1_test = prep1_email(mail)
    return final1_test


def email(final2_test):
    X_test = [sent2features_email(s) for s in final2_test]
    import pickle
    lin = 'tanuj28/cba_demo'
    crf2 = pickle.load(open(lin+"/"+'crf_model_email_id.p','rb'))
    y_pred = crf2.predict(X_test)
    return y_pred



def to_cap_cap(word, word1, word2):
    p = False
    if(word=="to" and word1.istitle() and word2.istitle()):
        p = True
    return p


# In[ ]:


def prev_word_reci(word):
    p = False
    if ((word == "to" or word == "at" or word == "attention" or word == "ATTN" or word == "Attn" or word=="att") ):
        p = True
    return p


# In[ ]:


def two_istitle(word, word1):
    p = False
    if (word.istitle() and word1.istitle()):
        p = True
    return p


# In[ ]:


def ner_stnfrd(word):
    ner_word =  False
    if word == 'PERSON':
        ner_word = True
    return ner_word


# In[ ]:


def word2features_reci(sent,i):
    word = sent[i][0]
    postag = sent[i][1]
    ner_tag = sent[i][2] 
    st_tag = sent[i][3]

    features = {
        'bias': 1.0,
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.islower()': word.islower(),
        'word.isdigit()': word.isdigit(),
        'word.isalpha()':word.isalpha(),
        'postag': postag,
        'word.shape()':shape(word),
        'word.person':ner_person(ner_tag),
        #'word[-3:]': word[-3:],     #last 3 lettets of each word
        #'word[-4:]': word[-4:],
        #'word[:3]': word[:3],     #First 3 lettets of each word
        #'word[:4]': word[:4],
        'word_stnfrd_loc': ner_stnfrd(st_tag),
        'st_tag':st_tag,
        
    }
    
    if i > 2:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        ner_tag1 = sent[i-1][2]
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        ner_tag2 = sent[i-2][2]
        word3 = sent[i-3][0]
        postag3 = sent[i-3][1]
        ner_tag3 = sent[i-3][2]
        st_tag1 =sent[i-1][3]
        st_tag2 =sent[i-2][3]
        st_tag3 =sent[i-3][3]
       
        features.update({
            
            '-1:word.person':ner_person(ner_tag1),
            
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.islower()': word1.islower(),
            '-1:postag': postag1,
            
            '-1:word.isdigit()':word1.isdigit(),  
            
            
            '-1:word.isalpha()':word1.isalpha(),  
            #'-1:word[-3:]': word1[-3:],
            #'-1:word[-4:]': word1[-4:],
            #'-1:word[:3]': word1[:3],
            #'-1:word[:4]': word1[:4],
            '-1:word.is_to': prev_word_reci( word1),
            '-1:word.shape()':shape(word1),
            '-1:word.grams':grams(word1,word),
            '-1:word.two_istitle':two_istitle(word,word1),
            
            '-1:word_stnfrd_loc': ner_stnfrd(st_tag1),
            '-1:st_tag':st_tag1,
            
            '-2:word.person':ner_person(ner_tag2),
            
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.islower()': word2.islower(),
            '-2:postag': postag2,
            
            #'-2:word[-3:]': word2[-3:],
            #'-2:word[-4:]': word2[-4:],
            #'-2:word[:3]': word2[:3],
            #'-2:word[:4]': word2[:4], 
            '-2:word.is_to': prev_word_reci(word2),
            '-2:word.isdigit()':word2.isdigit(),
            
            '-2:word.isalpha()':word2.isalpha(),
            '-2:word.grams':grams(word2,word1),
            '-2:word.shape()':shape(word2),
            
            '-2:word_stnfrd_loc': ner_stnfrd(st_tag2),
            '-2:st_tag':st_tag2,
            
            '-3:word.person':ner_person(ner_tag3),
            
             
            '-3:word.istitle()': word3.istitle(),
            '-3:word.isupper()': word3.isupper(),
            '-3:word.islower()': word3.islower(),
            '-3:postag': postag3,
            
            '-3:word.isdigit()':word3.isdigit(),
           
            '-3:word.isalpha()':word3.isalpha(),
            '-3:word.shape()':shape(word3),
            #'-3:word[-3:]': word3[-3:],
            #'-3:word[-4:]': word3[-4:],
            #'-3:word[:3]': word3[:3],
            #'-3:word[:4]': word3[:4],
            '-3:word.is_to': prev_word_reci(word3),
            '-3:word.grams':grams(word3,word2),
            '-3:word_stnfrd_loc': ner_stnfrd(st_tag3),
            '-3:st_tag':st_tag3,

        })
        
   
    elif i > 1:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        ner_tag1 = sent[i-1][2]
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        ner_tag2 = sent[i-2][2]
        st_tag1 = sent[i-1][3]
        st_tag2 = sent[i-2][3]
        
        features.update({
           
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.islower()': word1.islower(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:word.isdigit()':word1.isdigit(),  
            
            
            '-1:word.isalpha()':word1.isalpha(),
            '-1:word.person':ner_person(ner_tag1),
           
            '-1:word.shape()':shape(word1),
            #'-1:word[-3:]': word1[-3:],
            #'-1:word[-4:]': word1[-4:],
            #'-1:word[:3]': word1[:3],
            #'-1:word[:4]': word1[:4],
            '-1:word.is_to': prev_word_reci(word1),
            '-1:word.grams':grams(word1,word),
            
            
            '-1:word.two_istitle':two_istitle(word,word1),
            '-1:word_stnfrd_loc': ner_stnfrd(st_tag1),
            '-1:st_tag':st_tag1,
             
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.islower()': word2.islower(),
            '-2:word.person':ner_person(ner_tag2),
            
            '-2:postag': postag2,
            
            '-2:word.isdigit()':word2.isdigit(),
            
            '-2:word.isalpha()':word2.isalpha(),
            '-2:word.shape()':shape(word2),
            #'-2:word[-3:]': word2[-3:],
            #'-2:word[-4:]': word2[-4:],
            #'-2:word[:3]': word2[:3],
            #'-2:word[:4]': word2[:4],
            '-2:word.is_to': prev_word_reci(word2),
            '-2:word.grams':grams(word2,word1),
            '-2:word_stnfrd_loc': ner_stnfrd(st_tag2),
            '-2:st_tag':st_tag2,
                
        })
    
    elif i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        ner_tag1 = sent[i-1][2]
        st_tag1 = sent[i-1][3]
        
        features.update({
             
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.islower()': word1.islower(),
            '-1:postag': postag1,
            
            '-1:word.isdigit()':word1.isdigit(),  
            
            
            '-1:word.isalpha()':word1.isalpha(),
            '-1:word.shape()':shape(word1),
            '-1:word.person':ner_person(ner_tag1),
            
            #'-1:word[-3:]': word1[-3:],
            #'-1:word[-4:]': word1[-4:],
            #'-1:word[:3]': word1[:3],
            #'-1:word[:4]': word1[:4],
            '-1:word.is_to': prev_word_reci(word1),
            '-1:word.grams':grams(word1,word),
            
            '-1:word.two_istitle':two_istitle(word,word1),
            '-1:word_stnfrd_loc': ner_stnfrd(st_tag1),
            '-1:st_tag':st_tag1,
            })
            
       
    
    else:
        features['BOS'] = True

        
        
   
    if i < len(sent) -3:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        ner_tag1 = sent[i+1][2]
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        ner_tag2 = sent[i+2][2]
        word3 = sent[i+3][0]
        postag3 = sent[i+3][1]
        ner_tag3 = sent[i+3][2]
        
        st_tag1 = sent[i+1][3]
        st_tag2 = sent[i+2][3]
        st_tag3 = sent[i+3][3]
    
        features.update({
            
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.islower()': word1.islower(),
            '+1:postag': postag1,
            
            '+1:word.isdigit()':word1.isdigit(),
            '+1:word.person':ner_person(ner_tag1),
            
            
            
            '+1:word.isalpha()':word1.isalpha(),
            '+1:word.shape()':shape(word1),
            #'+1:word[-3:]': word1[-3:],
            #'+1:word[-4:]': word1[-4:],
            #'+1:word[:3]': word1[:3],
            #'+1:word[:4]': word1[:4],
            '+1:word.grams':grams(word,word1),
            '+1:word_stnfrd_loc': ner_stnfrd(st_tag1),
            '+1:st_tag':st_tag1,
            
            '+1:word.two_istitle':two_istitle(word,word1),
            
            '+2:word.isdigit()':word2.isdigit(),
            
            '+2:word.person':ner_person(ner_tag2),
            
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.islower()': word2.islower(),
            '+2:postag': postag2,
            
            
            '+2:word.isalpha()':word2.isalpha(),
            '+2:word.shape()':shape(word2),
            #'+2:word[-3:]': word2[-3:],
            #'+2:word[-4:]': word2[-4:],
            #'+2:word[:3]': word2[:3],
            #'+2:word[:4]': word2[:4],
            '+2:word.grams':grams(word1,word2),
            '+2:word_stnfrd_loc': ner_stnfrd(st_tag2),
            '+2:st_tag':st_tag2,
           
            '+3:word.istitle()': word3.istitle(),
            '+3:word.isupper()': word3.isupper(),
            '+3:word.islower()': word3.islower(),
            '+3:word.person':ner_person(ner_tag3),
            
            '+3:postag': postag3,
            
            '+3:word.isdigit()':word3.isdigit(),
            
            '+3:word.isalpha()':word3.isalpha(),
            '+3:word.shape()':shape(word3),
            #'+3:word[-3:]': word3[-3:],
            '+3:word[-4:]': word3[-4:],
            #'+3:word[:3]': word3[:3],
            '+3:word[:4]': word3[:4],
            '+3:word.grams':grams(word2,word3),
              
            
            '+3:word.to_cap_cap':to_cap_cap(word,word1, word2),
            '+3:word_stnfrd_loc': ner_stnfrd(st_tag3),
            '+3:st_tag':st_tag3,
            })
    
    
    elif i < len(sent)- 2:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        ner_tag1 = sent[i+1][2]
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        ner_tag2 = sent[i+1][2]
        st_tag1  = sent[i+1][3]
        st_tag2  = sent[i+2][3]
        
        
        features.update({
            
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.islower()': word1.islower(),
            '+1word.person':ner_person(ner_tag1),
            
            #'+1:word[-3:]': word1[-3:],
            #'+1:word[-4:]': word1[-4:],
            #'+1:word[:3]': word1[:3],
            #'+1:word[:4]': word1[:4],
            '+1:postag': postag1,
            
            '+1:word.isdigit()':word1.isdigit(),
            
            '+1:word.isalpha()':word1.isalpha(),
            '+1:word.shape()':shape(word1),
            
            '+1:word.grams':grams(word,word1),
            
            
            '+1:word.two_istitle':two_istitle(word,word1),
            '+1:word_stnfrd_loc': ner_stnfrd(st_tag1),
            '+1:st_tag':st_tag1,
            
            '+2:word.isdigit()':word2.isdigit(),
            
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.islower()': word2.islower(),
            '+2:word.person':ner_person(ner_tag2),
            
            '+2:postag': postag2,
            
            
            '+2:word.isalpha()':word2.isalpha(),
            '+2:word.shape()':shape(word2),
           # '+2:word[-3:]': word2[-3:],
            #'+2:word[-4:]': word2[-4:],
            #'+2:word[:3]': word2[:3],
           # '+2:word[:4]': word2[:4],
            '+2:word.grams':grams(word1,word2),
            '+2:word.to_cap_cap':to_cap_cap(word,word1, word2),
            '+2:word_stnfrd_loc': ner_stnfrd(st_tag2),
            '+2:st_tag':st_tag2,
            
        })        
    
    elif i < len(sent) -1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        ner_tag1 = sent[i+1][2]
        st_tag1 = sent[i+1][3]
    
        features.update({
            
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.islower()': word1.islower(),
            '+1:postag': postag1,
            
            '+1:word.isdigit()':word1.isdigit(),
            '+1:word.person':ner_person(ner_tag1),
            
            
           
            '+1:word.isalpha()':word1.isalpha(),
            '+1:word.shape()':shape(word1),
            #'+1:word[-3:]': word1[-3:],
           # '+1:word[-4:]': word1[-4:],
            #'+1:word[:3]': word1[:3],
           # '+1:word[:4]': word1[:4],
            '+1:word.grams':grams(word,word1),
            
            '+1:word.two_istitle':two_istitle(word,word1),
            '+1:word_stnfrd_loc': ner_stnfrd(st_tag1),
            '+1:st_tag':st_tag1,
                   
            })
    
    
    else:
        features['EOS'] = True

    return features


# In[ ]:


def sent2features_reci(sent):
            return [word2features_reci(sent,i) for i in range(len(sent))]


# In[ ]:


def prep_reci(final1_test):
    final2_test = prep2_reci(final1_test)
    return final2_test

def make_fin1_rec(words):
    final1_test = prep1_reci(words)
    return final1_test


# In[ ]:


def reci(final2_test):
    X_test = [sent2features_reci(s) for s in final2_test]
    import pickle
    lin = 'tanuj28/cba_demo'
    crf2 = pickle.load(open(lin+"/"+'crf_model_reci_st.p','rb'))
    y_pred = crf2.predict(X_test)
    
    ind = [it for it,v in enumerate(y_pred[0]) if v == 'Recipient']
    reci_proba = np.mean([crf2.predict_marginals_single(y_pred[0])[it]["Recipient"] for it in ind])
    return y_pred, reci_proba


def to_cap(word1, word):
    p = False
    if ((word1 == "to" and word.istitle()) or (word1 == "to" and word == 'BSB:')):
        p = True
    return p


# In[ ]:


def BSB(word, word1):
    p =  False
    if (word == 'BSB:' and word1.isdigit()):
        p = True
    return p


# In[ ]:


def ner_stnfrd_add(word):
    ner_word =  False
    if word == 'LOCATION' or word == 'ORGANISATION':
        ner_word = True
    return ner_word


# In[ ]:


def ner_loc(word):
    ner_word = False
    
    if word == 'B-LOCATION' or word == 'I-LOCATION':
        ner_word = True
        
    return ner_word


# In[ ]:


def word2features_add(sent,i):
    word = sent[i][0]
    postag = sent[i][1]
    ner_tag = sent[i][2] 
    st_tag = sent[i][3]


    features = {
        'bias': 1.0,
        
        'word.isupper()': word.isupper(),
        'word.islower()': word.islower(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isalpha()':word.isalpha(),
        'postag': postag,
        'word.shape()':shape(word),
        'word.loc':ner_loc(ner_tag),
        'word_stnfrd_loc': ner_stnfrd_add(st_tag),
        'st_tag':st_tag,
        
    }
    
    if i > 2:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        ner_tag1 = sent[i-1][2]
        st_tag1 = sent[i-1][3]
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        ner_tag2 = sent[i-2][2]
        st_tag2 = sent[i-2][3]
        word3 = sent[i-3][0]
        postag3 = sent[i-3][1]
        ner_tag3 = sent[i-3][2]
        st_tag3 = sent[i-3][3]
    
        features.update({
            
            '-1:word.person':ner_loc(ner_tag1),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.islower()': word1.islower(),
            '-1:postag': postag1,
            '-1:word.isdigit()':word1.isdigit(),  
            '-1:word.isalpha()':word1.isalpha(),  
            '-1:word.is_to': prev_word(word1),
            '-1:word.shape()':shape(word1),
            '-1:word.grams':grams(word1,word),
            '-1:word.to_cap':to_cap(word1,word),
            '-1:word_stnfrd_loc': ner_stnfrd_add(st_tag1),
            '-1:st_tag':st_tag1,
            
            
            '-2:word.person':ner_loc(ner_tag2),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.islower()': word2.islower(),
            '-2:postag': postag2, 
            #'-2:word.is_to': prev_word(word2),
            '-2:word.isdigit()':word2.isdigit(),
            '-2:word.isalpha()':word2.isalpha(),
            '-2:word.grams':grams(word2,word1),
            '-2:word.shape()':shape(word2),
            
            '-2:word_stnfrd_loc': ner_stnfrd_add(st_tag2),
            '-2:st_tag':st_tag2,
            
            
            '-3:word.person':ner_loc(ner_tag3),
            '-3:word.istitle()': word3.istitle(),
            '-3:word.isupper()': word3.isupper(),
            '-3:word.islower()': word3.islower(),
            '-3:postag': postag3,
            '-3:word.isdigit()':word3.isdigit(),
            '-3:word.isalpha()':word3.isalpha(),
            '-3:word.shape()':shape(word3),
            #'-3:word.is_to': prev_word(word3),
            '-3:word.grams':grams(word3,word2),
            '-3:word_stnfrd_loc': ner_stnfrd_add(st_tag3),
            '-3:st_tag':st_tag3,

        })
        
   
    elif i > 1:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        ner_tag1 = sent[i-1][2]
        word2 = sent[i-2][0]
        postag2 = sent[i-2][1]
        ner_tag2 = sent[i-2][2]
        st_tag1 = sent[i-1][3]
        st_tag2 = sent[i-2][3]
        
        features.update({
             
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.islower()': word1.islower(),
            '-1:postag': postag1,
            '-1:word.isdigit()':word1.isdigit(), 
            '-1:word.isalpha()':word1.isalpha(),
            '-1:word.person':ner_loc(ner_tag1),
            '-1:word.shape()':shape(word1),
            '-1:word.is_to': prev_word(word1),
            '-1:word.grams':grams(word1,word),
            '-1:word.to_cap':to_cap(word1,word),
            '-1:word_stnfrd_loc': ner_stnfrd_add(st_tag1),
            '-1:st_tag':st_tag1,
            
            
            
            
             
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.islower()': word2.islower(),
            '-2:word.person':ner_loc(ner_tag2),
            '-2:postag': postag2,
            '-2:word.isdigit()':word2.isdigit(),
            '-2:word.isalpha()':word2.isalpha(),
            '-2:word.shape()':shape(word2),
            #'-2:word.is_to': prev_word(word2),
            '-2:word.grams':grams(word2,word1),
              
            '-2:word_stnfrd_loc': ner_stnfrd_add(st_tag2),
            '-2:st_tag':st_tag2,
        })
    
    elif i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        ner_tag1 = sent[i-1][2]
        st_tag1 = sent[i-1][3]
        
        features.update({
            
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.islower()': word1.islower(),
            '-1:postag': postag1,
            '-1:word.isdigit()':word1.isdigit(),
            '-1:word.isalpha()':word1.isalpha(),
            '-1:word.shape()':shape(word1),
            '-1:word.person':ner_loc(ner_tag1),
            '-1:word.is_to': prev_word(word1),
            '-1:word.grams':grams(word1,word),
            '-1:word.to_cap':to_cap(word1,word),
            
            '-1:word_stnfrd_loc': ner_stnfrd_add(st_tag1),
            '-1:st_tag':st_tag1,
            
            'first_word': True,
            
            })
            
       
    
    else:
        features['BOS'] = True

        
        
   
    if i < len(sent) -3:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        ner_tag1 = sent[i+1][2]
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        ner_tag2 = sent[i+2][2]
        word3 = sent[i+3][0]
        postag3 = sent[i+3][1]
        ner_tag3 = sent[i+3][2]
        st_tag1 = sent[i+1][3]
        st_tag2 =sent[i+2][3]
        st_tag3 =sent[i+3][3]
    
        features.update({
            
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.islower()': word1.islower(),
            '+1:postag': postag1,
            '+1:word.isdigit()':word1.isdigit(),
            '+1:word.person':ner_loc(ner_tag1),
            '+1:word.isalpha()':word1.isalpha(),
            '+1:word.shape()':shape(word1),
            '+1:word.grams':grams(word,word1),
            '+1:word.BSB':BSB(word, word1),
            '+1:word_stnfrd_loc': ner_stnfrd_add(st_tag1),
            '+1:st_tag':st_tag1,
            
            
            '+2:word.isdigit()':word2.isdigit(),
            '+2:word.person':ner_loc(ner_tag2),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.islower()': word2.islower(),
            '+2:postag': postag2,
            '+2:word.isalpha()':word2.isalpha(),
            '+2:word.shape()':shape(word2),
            '+2:word.grams':grams(word1,word2),
            '+2:word.BSB':BSB(word, word2),
            '+2:word_stnfrd_loc': ner_stnfrd_add(st_tag2),
            '+2:st_tag':st_tag2,
            
            
            '+3:word.istitle()': word3.istitle(),
            '+3:word.isupper()': word3.isupper(),
            '+3:word.islower()': word3.islower(),
            '+3:word.person':ner_loc(ner_tag3),
            '+3:postag': postag3,
            '+3:word.isdigit()':word3.isdigit(),
            '+3:word.isalpha()':word3.isalpha(),
            '+3:word.shape()':shape(word3),
            '+3:word.grams':grams(word2,word3),
            '+3:word_stnfrd_loc': ner_stnfrd_add(st_tag3),
            '+3:st_tag':st_tag3,
            })
    
    
    elif i < len(sent)- 2:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        ner_tag1 = sent[i+1][2]
        word2 = sent[i+2][0]
        postag2 = sent[i+2][1]
        ner_tag2 = sent[i+1][2]
        st_tag1 = sent[i+1][3]
        st_tag2 =sent[i+2][3]
        
        features.update({
            
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.islower()': word1.islower(),
            '+1word.person':ner_loc(ner_tag1),
            '+1:postag': postag1,
            '+1:word.isdigit()':word1.isdigit(),
            '+1:word.isalpha()':word1.isalpha(),
            '+1:word.shape()':shape(word1),
            '+1:word.grams':grams(word,word1),
            '+1:word.BSB':BSB(word, word1),
            '+1:word_stnfrd_loc': ner_stnfrd_add(st_tag1),
            '+1:st_tag':st_tag1,
            
            
            '+2:word.isdigit()':word2.isdigit(),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.islower()': word2.islower(),
            '+2:word.person':ner_loc(ner_tag2),
            '+2:postag': postag2,
            '+2:word.isalpha()':word2.isalpha(),
            '+2:word.shape()':shape(word2),
            '+2:word.grams':grams(word1,word2),
            '+2:word.BSB':BSB(word, word2),  
            '+2:word_stnfrd_loc': ner_stnfrd_add(st_tag2),
            '+2:st_tag':st_tag2,
            
        })        
    
    elif i < len(sent) -1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        ner_tag1 = sent[i+1][2]
        st_tag1 = sent[i+1][3]
        features.update({
            
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.islower()': word1.islower(),
            '+1:postag': postag1,
            '+1:word.isdigit()':word1.isdigit(),
            '+1:word.person':ner_loc(ner_tag1),
            '+1:word.isalpha()':word1.isalpha(),
            '+1:word.shape()':shape(word1),
            '+1:word.grams':grams(word,word1),
            '+1:word.BSB':BSB(word, word1), 
            '+1:word_stnfrd_loc': ner_stnfrd_add(st_tag1),
            '+1:st_tag':st_tag1,
            })
    
    
    else:
        features['EOS'] = True

    return features


# In[ ]:


def sent2features_add(sent):
    return [word2features_add(sent,i) for i in range(len(sent))]


# In[ ]:


def prep_add(final1_test):
    final2_test = prep2_add(final1_test)
    return final2_test


def make_fin1_add(words):
    final1_test = prep1_add(words)
    return final1_test


def add(final2_test):
    X_test = [sent2features_add(s) for s in final2_test]
    import pickle
    lin = 'tanuj28/cba_demo'
    crf3 = pickle.load(open(lin+"/"+'crf_model_addrs_st.p','rb'))
    y_pred = crf3.predict(X_test)
    return y_pred

def sentences_to_words( raw_sentences ):
    #sentences_text =BeautifulSoup(raw_sentences, "lxml").get_text()
    sentences_text = raw_sentences.replace(".", "")
    letters_only = re.sub("[^a-zA-Z]", " ", sentences_text) 
    lowercs = letters_only.lower()
    lowercs = re.sub('intenral','internal', lowercs)
    lowercs = lowercs.replace("  ", " ")
    words_1 = lowercs.split()
    stem_words = " ".join(words_1)
    lowercs = stem_words
    words_1 = lowercs.split()
    stem_words = " ".join(words_1)
    lowercs = stem_words
    words =lowercs.split()
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join( meaningful_words ))

def prep1_method (words):
    clean_test_sentences = []
    clean_test_sentences.append( sentences_to_words(words))
    return clean_test_sentences


# In[ ]:


def method(clean_test_sentences):
    import pickle
    import pickle

    lin = 'tanuj28/cba_demo'
    vect = pickle.load(open(lin+"/"+'vectorizer_method.p','rb'))
    test_data_features = vect.transform(clean_test_sentences)
    test_data_features = test_data_features.toarray()


    clf4 = pickle.load(open(lin+"/"+'SVM_method_svc.p','rb'))
    y_forc = clf4.predict(test_data_features)

    encode = pickle.load(open(lin+"/"+'le_methd.p','rb'))
    out = encode.inverse_transform([y_forc])
    return out



from dateutil.parser import parse

def is_date(string):
    try: 
        parse(string)
        return True
    except ValueError:
        return False


from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

import pickle

def sentences_to_words_intent(raw_sentence):
    #sentences_text = raw_sentences.replace(".", "")
    raw_sentence=re.sub('intenral','internal', raw_sentence)
    
    clean_sentence = re.sub(' +', ' ',raw_sentence)
    #print(clean_sentence)
    
    word = nltk.regexp_tokenize(clean_sentence, pattern=r'[^\w.@/]', gaps=True)
    #print (word)
    words = []
    
    for ww in word:
        
        if "/" in ww:
            if len(ww) == 2 or len(ww) ==3:
                dd = ww.split('/')
                for i in range(len(dd)):
                    if dd[i] != '':
                        words.append(dd[i])
                        
            elif is_date(ww) == False:
                
                dd = ww.split('/')
                for i in range(len(dd)):
                    if dd[i] != '':
                        words.append(dd[i])
                        
            else:
                words.append(ww)
        else:
            words.append(ww)
        
                
    
    #print (words)
    tokensList = [w.strip() for w in words]
    
    sent_tokenize_list = sent_tokenize(" ".join(tokensList).strip())
    lemmatized_sent = []
    for i in range(len(sent_tokenize_list)):
        final_word1 = []
        if sent_tokenize_list[i].endswith(suffix):
            sent = sent_tokenize_list[i][:len(sent_tokenize_list[i])-1]
            final_word = sent.split()
            
            for cc in final_word:
                final_word1.append(cc)
                
                
            pos_tag = nltk.pos_tag(final_word1)
            for i in range(len(pos_tag)):
                lemmatized_sent.append(lmtzr.lemmatize(pos_tag[i][0].lower().strip(),get_wordnet_pos(pos_tag[i][1])))
    
            
            
            
        else:
            sent = sent_tokenize_list[i]
            final_word = sent.split()
            for cc in final_word:
                final_word1.append(cc)
                        
            pos_tag = nltk.pos_tag(final_word1)
            for i in range(len(pos_tag)):
                lemmatized_sent.append(lmtzr.lemmatize(pos_tag[i][0].lower().strip(),get_wordnet_pos(pos_tag[i][1])))
    
    
    lemmatized_sent= " ".join(lemmatized_sent).strip()
    lowercaseTokens =lemmatized_sent.lower().split() 
    
    
    #print(final_word1)
    #meaningful_words = [w for w in lowercaseTokens if not w in stop]
    
    return( " ".join(lowercaseTokens).strip())

def recognize_entity(mail):
    category, met_prob = recognize_method(mail)
    recipient, reci_prob, ln_reci = recognize_recipient(mail)
    address, add_prob, ln_add = recognize_address(mail)
    email_id, email_prob, ln_email = recognize_email_id(mail)
    return category, met_prob, recipient, reci_prob, ln_reci, email_id,email_prob, ln_email, address, add_prob, ln_add

data_mulilabel_multiclass_name = {
            0:'Trust Investigation',
            1:'Seek Review',
            2:'Seek Approval',
            3:'No Intent',
            4:'Documentation',
            5:'Seek Confirmation',
            6:'Seek Approval AND Seek Review',
            7:'Home Loan Switch'
    
}



def recognize_intent(mail):
    if len(mail) != " ":
        
        test_sent = sentences_to_words_intent(mail)
        with open('tanuj28/cba_demo/vocab_train_test.p', 'rb') as handle:
            vocab = pickle.load(handle)
    
        with open('tanuj28/cba_demo/logistic_regression_model_train_test.p', 'rb') as handle:
            model = pickle.load(handle)
    
    
        count_vect_POS_D = CountVectorizer(binary=True,vocabulary=vocab.Features,ngram_range=(1,2))
        X_counts = count_vect_POS_D.fit_transform([test_sent])
    #X_counts.shape
    
    
        pred_lr = model.predict(X_counts)
        pred_lr_prob=model.predict_proba(X_counts)
    #print (pred_lr_prob)
    

            
            
        if pred_lr[0] != 6:
            
            first_intent = data_mulilabel_multiclass_name[pred_lr[0]]
            first_intent_prob = round(pred_lr_prob[0][pred_lr[0]],2)
            second_intent = 'Absent'
            second_intent_prob = 0
            scor = 1
            return  scor, first_intent,first_intent_prob,second_intent,second_intent_prob
     
    
        else:
            label = data_mulilabel_multiclass_name[6].split('AND')
            first_intent = label[0]
            first_intent_prob = round(pred_lr_prob[0][pred_lr[0]],2)
            second_intent = label[1]
            second_intent_prob = round(pred_lr_prob[0][pred_lr[0]],2)
            scor = 2
            return  scor, first_intent,first_intent_prob,second_intent,second_intent_prob
    
 
    else:
        return None
    
    
    
    
def recognize_email_id(mail):
    final1_test = make_fin1_email(mail)
    final2_test = prep_email(final1_test)
    X_test = [sent2features_email(itr) for itr in final2_test]
    y_pred = email(final2_test)
    import pickle
    lin = 'tanuj28/cba_demo/CBA_GUI_test_data'
    crf = pickle.load(open(lin+"/"+'crf_model_email_id.p','rb'))
    y = [i for i, elem in enumerate(y_pred[0]) if 'Email Address' in elem]
    if (len(y) == 0):
        out = "Not present"
    else:
        x = ";"
        out = []
        for j in range(len(y)):
            ind = y[j]
            q = final2_test[0][ind][0]
            if (j == 0):
                out.append(q)
            else:
                ind_prev = y[j-1]
                if (ind == ind_prev + 1):
                    out.append(q)
                else:
                    out.append(x)
                    out.append(q)
            if (j == len(y) - 1):
                out.append(x)
        out = " ".join(out)
        
    email_nam = []
    email_proba = []
    if (out == "Not present"):
        email_nam.append("Absent")
        email_proba.append(0)
        ln_email = 0   
    else:   
        p = 0
        rec_name = str(out)
        str_l = rec_name.split(";")
        s_out = ""
        for s in str_l:
            s_out += s
            s1 = s.split()
            max_s1 = 0
            if(len(s1)>0):
                for s2 in s1:
                    ind = [it for it, v in enumerate(final1_test[p]) if v == s2]
                    recip_val = [crf.predict_marginals_single(X_test[p])[it]["Email Address"] for it in ind]
                    max_recip_s2 = max(recip_val)
                    max_s1 += max_recip_s2
                max_s1 = max_s1/len(s1)
                max_s1 = round(max_s1, 2)
                email_nam.append(s)
                email_proba.append(str(max_s1))
        ln_email = len(email_proba)
    return email_nam, email_proba, ln_email



def recognize_recipient(mail):
    final1_test = make_fin1_rec(mail)
    final2_test = prep_reci(final1_test)
    X_test = [sent2features_reci(itr) for itr in final2_test]
    y_pred, reci_proba = reci(final2_test)
    import pickle
    lin = 'tanuj28/cba_demo/CBA_GUI_test_data'
    crf2 = pickle.load(open(lin+"/"+'crf_model_reci_st.p','rb'))
    y = [i for i, elem in enumerate(y_pred[0]) if 'Recipient' in elem]
    if (len(y) == 0):
        out = "Not present"
    else:
        x = ";"
        out = []
        for j in range(len(y)):
            ind = y[j]
            q = final2_test[0][ind][0]
            if (j == 0):
                out.append(q)
            else:
                ind_prev = y[j-1]
                if (ind == ind_prev + 1):
                    out.append(q)
                else:
                    out.append(x)
                    out.append(q)
        out = " ".join(out)
    reci_nam = []
    reci_proba = []
    if (out == "Not present"):
        reci_nam.append("Absent")
        reci_proba.append(0)
        ln_reci = 0   
    else:   
        p = 0
        rec_name = str(out)
        str_l = rec_name.split(";")
        s_out = ""
        for s in str_l:
            s_out += s
            s1 = s.split()
            max_s1 = 0
            if(len(s1)>0):
                for s2 in s1:
                    ind = [it for it, v in enumerate(final1_test[p]) if v == s2]
                    recip_val = [crf2.predict_marginals_single(X_test[p])[it]["Recipient"] for it in ind]
                    max_recip_s2 = max(recip_val)
                    max_s1 += max_recip_s2
                max_s1 = max_s1/len(s1)
                max_s1 = round(max_s1, 2)
                reci_nam.append(s)
                reci_proba.append(str(max_s1))
        ln_reci = len(reci_proba)
        
        for i in range(len(reci_nam)):
            reci_nam[i] = reci_nam[i].rstrip(' ')
            reci_nam[i] = reci_nam[i].lstrip(' ')

        l1 = reci_nam
        prob_l = reci_proba

        set_list = list(set(l1))
        prob_list = []
        for l in set_list:
            ind =[it for it,v in enumerate(l1) if v==l]
            prob_val =[prob_l[it] for it in ind]
            prob_list.append(max(prob_val))
        ln_reci = len(prob_list)
        reci_nam = set_list
        reci_proba = prob_list
    return reci_nam, reci_proba, ln_reci

def recognize_address(mail):
    final1_test = make_fin1_add(mail)
    final2_test = prep_add(final1_test)
    X_test = [sent2features_add(itr) for itr in final2_test]
    y_pred = add(final2_test)
    import pickle
    lin = 'tanuj28/cba_demo/CBA_GUI_test_data'
    crf3 = pickle.load(open(lin+"/"+'crf_model_addrs_st.p','rb'))
    y_pred = crf3.predict(X_test)
    y = [i for i, elem in enumerate(y_pred[0]) if 'Address' in elem]
    if (len(y) == 0):
        out = "Not present"
    else:
        x = ";"
        out = []
        for j in range(len(y)):
            ind = y[j]
            q = final2_test[0][ind][0]
            if (j == 0):
                out.append(q)
            else:
                ind_prev = y[j-1]
                if (ind == ind_prev + 1):
                    out.append(q)
                else:
                    out.append(x)
                    out.append(q)
            if (j == len(y) - 1):
                out.append(x)
        out = " ".join(out)
     
    addss = []
    add_proba = []
    if (out == "Not present"):
        addss.append("Absent")
        add_proba.append(0)
        ln_add = 0
    else:
        p = 0
        rec_name = str(out)
        str_l = rec_name.split(";")
        s_out = ""
        max_s1 = 0
        addss = []
        add_proba = []
        for s in str_l:
            s_out += s
            s1 = s.split()
            if(len(s1)>0):
                for s2 in s1:
                    ind = [it for it, v in enumerate(final1_test[p]) if v == s2]
                    recip_val = [crf3.predict_marginals_single(X_test[p])[it]["Address"] for it in ind]
                    max_recip_s2 = max(recip_val)
                    max_s1 += max_recip_s2
                max_s1 = max_s1/len(s1)
                max_s1 = round(max_s1, 2)
                addss.append(s)
                add_proba.append(str(max_s1))
        ln_add = len(add_proba)
    return addss, add_proba, ln_add


def recognize_method(mail):
        clean_test_sentences = prep1_method(mail)
        y_pred = method(clean_test_sentences)
        out_method = y_pred[0]
        import pickle

        lin = 'tanuj28/cba_demo/CBA_GUI_test_data'
        vect = pickle.load(open(lin+"/"+'vectorizer_method.p','rb'))
        test_data_features = vect.transform(clean_test_sentences)
        test_data_features = test_data_features.toarray()


        clf4 = pickle.load(open(lin+"/"+'SVM_method_svc.p','rb'))
        max_prob = max(clf4.predict_proba(test_data_features)[0])
        max_prob = round(max_prob, 2)
        a = str(out_method[0])
        b = " \t Probability_Score:"
        string_length=len(b)+10    # will be adding 10 extra spaces
        c = b.rjust(string_length)
        d = a + c + str(max_prob) + " ; "
        if max_prob > 0.5:
            return a, str(max_prob)
        else:
            a = str("Absent")
            max_prob = 0
            return a, str(max_prob)


app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/number_in', methods=['GET', 'POST'])
def number_in():
    mail = None
    if request.method == 'POST':
        num = int(request.form['number'])
        if num >= len(list_a):
            mail = "You are out of the range. Please Enter a valid number. :-("
        else:
            mail = list_a[num]
    return render_template('number_in.html', mail = mail)


@app.route('/result_num/<path:mail>', methods=['GET','POST'])
def result_num(mail):
    flag = None
    if request.method == 'GET':
        flag = 1
        category,met_prob, email_id, recipient, address = recognize_entity(mail)
        return render_template('result_num.html', category = category,met_prob = met_prob, recipient = recipient, email_id = email_id, address = address, mail = mail, flag = flag)
    elif request.method =='POST':
        num = int(request.form['number'])
        if num >= len(list_a):
            mail = "You are out of the range. Please Enter a valid number. :-("
        else:
            mail = list_a[num]
        category, met_prob, email_id, recipient, address = recognize_entity(mail)
        return render_template('result_num.html', category = category,met_prob = met_prob, recipient = recipient, email_id = email_id, address = address, mail = mail, flag = flag)    
        
@app.route('/mail_in', methods=['GET', 'POST'])
def mail_in():
    mail = None
    if request.method == 'POST':
        mail = request.form['mail']
    return render_template('mail_in.html', mail = mail)

@app.route('/result_mail/', methods=['GET','POST'])
def result_mail():
    mail = None; entity = None; intent = None; flag = None
    if request.method == 'GET':
        return render_template('result_mail.html', mail = mail, flag = flag)
    elif request.method =='POST':
        flag = 1
        if request.form["submit"] == "Entity":
            mail = request.form['mail']
            if mail.split() == []:
                mail = ""
            entity = 1
            if mail != "":
                category,met_prob, recipient,reci_prob, ln_reci, email_id,email_prob, ln_email, address, add_prob, ln_add = recognize_entity(mail)
                return render_template('result_mail.html', category = category,met_prob = met_prob, email_id = email_id,email_prob = email_prob, ln_email = ln_email, recipient = recipient, reci_prob = reci_prob, ln_reci = ln_reci, address = address, add_prob = add_prob, ln_add = ln_add, mail = mail, entity = entity, intent = intent, flag = flag) 
            else:
                return render_template('result_mail.html', flag = flag)
        elif request.form["submit"] == "Intent":
            mail = request.form['mail']
            if mail.split() == []:
                mail = ""
            intent = 1
            if mail != "":
                num_intent, first_intent , first_intent_prob, second_intent, second_intent_prob = recognize_intent(mail)
                return render_template('result_mail.html', num_intent = num_intent, first_intent = first_intent, first_intent_prob = first_intent_prob,  second_intent = second_intent, second_intent_prob = second_intent_prob, mail = mail, entity = entity, intent = intent, flag = flag)
            else:
                return render_template('result_mail.html', flag = flag)


if __name__ == '__main__':
    app.run(use_reloader=True, debug=True, host = '127.0.0.1',port = 5075)