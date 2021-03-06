#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
os.system('pip install unidecode')
os.system('pip install gensim')
os.system('pip install flashtext')
os.system('pip install pyspark')
os.system('pip install tabulate')
# In[2]:


import json
import re
import traceback 
from pyspark.sql import SparkSession
from elasticsearch import Elasticsearch, helpers
import requests, json, csv
from nltk.corpus import stopwords
import string
import unidecode
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import nltk
import pandas as pd
from flashtext import KeywordProcessor
import spacy
import pytextrank
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
from nltk.stem import WordNetLemmatizer
import pprint
from IPython.display import display
from tabulate import tabulate


# In[3]:


s_top_words = set(stopwords.words('english'))


# ## Data loading

# In[4]:


def load_data(path = 'metadata.csv'):
    spark = SparkSession     .builder     .appName("ElasticSpark-1")     .config("spark.driver.extraClassPath", "/path/elasticsearch-hadoop-7.6.2/dist/elasticsearch-spark-20_2.11-7.6.2.jar")     .config("spark.es.port","9200")     .config("spark.driver.memory", "8G")     .config("spark.executor.memory", "12G")     .getOrCreate()
    metadata_df = spark.read.csv(path, multiLine=True, header=True)
    metadata_df.show(1)
    metadata_df = metadata_df.select("*").limit(1000)
    metadata_df.show(3)
    metadata_table = metadata_df.toPandas()
    print("Data loaded.")
    print(metadata_table)
    metadata_table["pr_title"]=metadata_table["title"]
    metadata_table["pr_abstract"]=metadata_table["abstract"]
    return metadata_table


# In[5]:


def create_index(es_index="covid"):
    res = requests.get('http://localhost:9200')
    print (res.content)
    es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])
    mapping = {
        'settings':{
            'number_of_shards': 1, 
            'number_of_replicas': 1
        },
        'mappings': {
            'properties': {
                'cord_uid': {
                    'index': 'false', 
                    'type': 'text'
                },
                'sha': {
                    'index': 'true', 
                    'type': 'text'
                },
                'source_x': {
                    'index': 'true', 
                    'type': 'text'
                },
                'title': {
                    'index': 'true',
                    'type': 'text', 
                    'similarity': 'BM25'
                },
                'pr_title': {
                    'index': 'true',
                    'type': 'text', 
                    'similarity': 'BM25'
                },
                'doi': {
                    'index': 'true', 
                    'type': 'text'
                },
                'pmcid': {
                    'index': 'true',
                    'type': 'text'
                },
                'license': {
                    'index': 'true',
                    'type': 'text'
                },
                'abstract': {
                    'index': 'true',
                    'type': 'text',
                    'similarity': 'BM25'
                },
                'pr_abstract': {
                    'index': 'true',
                    'type': 'text',
                    'similarity': 'BM25'
                },
                'publish_time': {
                    'index': 'true', 
                    'type': 'text'
                },
                'authors': {
                    'index': 'true',
                    'type': 'text'
                },
                'journal': {
                    'index': 'true',
                    'type': 'text'
                },
                'who_covidence_id': {
                    'index': 'true',
                    'type': 'text'
                },
                'arxiv_id': {
                    'index': 'true',
                    'type': 'text'
                },
                'pdf_json_files': {
                    'index': 'true',
                    'type': 'text'
                },
                'pmc_json_files': {
                    'index': 'true', 
                    'type': 'text'
                },
                'url': {
                    'index': 'true',
                    'type': 'text'
                },
                's2_id': {
                    'index': 'true', 
                    'type': 'text'
                }
             }
         }
    }
    if es.indices.exists(es_index):
        es.indices.delete(es_index) 

    es.indices.create(index=es_index,body=mapping)
    return es


# ## Sentence Splitting, Tokenization and Normalization

# In[6]:


class TextNormalizer:
    def __init__(self):
        self.punctuation_table = str.maketrans('','',string.punctuation)

    def normalize_text(self,text):
        if text==None:
            return None
        try: 
            normalized_sentences = []
            text = re.sub(' +',' ', text)
            text = unidecode.unidecode(text)
            text = text.lower()
            sentences = sent_tokenize(text)
        except:
            print("ERROR:", text)
            traceback.print_exc()
            return None
        
        for sentence in sentences:
            #remove punctuation
            sentence=re.sub("["+string.punctuation+"\d*]"," ",sentence)
            #strip leading/trailing whitespace
            sentence = sentence.strip()
            words = word_tokenize(sentence)
            new_sentence = ' '.join(words)
            normalized_sentences.append(new_sentence)
        return normalized_sentences


# In[7]:


def normalize_table(metadata_table):
    normaliser = TextNormalizer()
    
    table_to_process=metadata_table[["pr_title","pr_abstract"]]
    table_to_process["pr_title"]=table_to_process["pr_title"].apply(lambda x: normaliser.normalize_text(x))
    table_to_process["pr_abstract"]=table_to_process["pr_abstract"].apply(lambda x: normaliser.normalize_text(x))
    
    for i in range(0, len(table_to_process)):
        metadata_table.loc[i,"pr_title"] = table_to_process.loc[i,"pr_title"]
        metadata_table.loc[i,"pr_abstract"] = table_to_process.loc[i,"pr_abstract"]
    return metadata_table


# ## Selecting key words

# In[8]:


def remove_stop_words(text):
    if text==None:
        return
    for index,sentence in enumerate(text):
        sentence = sentence.split(" ")
        sentence = [word for word in sentence if word not in s_top_words and len(word)>2]
        sentence=" ".join(sentence)
        text[index]=sentence
    return text


# In[9]:


def get_words_corpus(table):
    words_corpus=[]
    for i in range(0, len(table)):
        row=table.loc[i]
        title_sentences = row["pr_title"]
        abstract_sentences = row["pr_abstract"]
        
        if title_sentences!=None:
            for i in range(0,len(title_sentences)):
                words_corpus.extend(title_sentences[i].split())
                
        if  abstract_sentences!=None:
            for i in range(0,len(abstract_sentences)):
                words_corpus.extend(abstract_sentences[i].split())
    return words_corpus
        


# In[10]:


def get_keywords_by_textrank(sentences):
    if sentences==None:
        return None
    keywords=dict()
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe("textrank", last=True)
    doc = nlp(" ".join(sentences))

    # examine the top-ranked phrases in the document

    for p in doc._.phrases:
        if p.rank>=0.05:
            keywords[p.text]=p.rank
    #         print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
    #         print(p.text)
    return keywords


# In[11]:


def extract_keywords(text,keyword_processor):
    sentences=[]
    if text==None:
        return None
    for i in range(0, len(text)):
        keywords_found = keyword_processor.extract_keywords(text[i])
        sentences.append(" ".join(keywords_found))
    return sentences
    


# In[12]:


def merge_two_keywords_methods(sentences, text_rank_key_word_processor, frequent_key_words_processor):
    if sentences==None:
        return None
    text_rank_version = extract_keywords(sentences,text_rank_key_word_processor)
    frequent_key_words_version = extract_keywords(sentences,frequent_key_words_processor)
    intersect = set(frequent_key_words_version) - set(text_rank_version)

    merged_version = text_rank_version + list(intersect)
    return merged_version


# In[13]:


def retain_best_tf_idf_keywords(sentences, index, tfIdf,tfIdfVectorizer):
    if sentences==None:
        return None
    tf_idf_keyword_processor = KeywordProcessor()
    df = pd.DataFrame(tfIdf[index].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF_IDF"])
    df = df.sort_values('TF_IDF', ascending=False)
    df = df[df.TF_IDF>0.09]
    tf_idf_dict=df.T.to_dict('list')    
    for keyword in tf_idf_dict.keys():
        parts = " ".join(keyword.split("_"))
        tf_idf_keyword_processor.add_keyword(keyword,parts)
    sentences = extract_keywords(sentences,tf_idf_keyword_processor)
    return sentences


# In[14]:


def select_best_keywords(metadata_table):
    table_to_process=metadata_table[["pr_title","pr_abstract"]]
    table_to_process["pr_title"]=table_to_process["pr_title"].apply(lambda x: remove_stop_words(x))
    table_to_process["pr_abstract"]=table_to_process["pr_abstract"].apply(lambda x: remove_stop_words(x))

    print("Text Data after removing of stop-words")
    print(table_to_process)

    words_corpus=get_words_corpus(table_to_process)
    print(len(words_corpus))

    dist = nltk.FreqDist(words_corpus) #Creating a distribution of words' frequencies
    grams=dist.most_common(1000) #Obtaining the most frequent words
    bigrams = nltk.collocations.BigramAssocMeasures()
    trigrams = nltk.collocations.TrigramAssocMeasures()

    bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(words_corpus)
    trigramFinder = nltk.collocations.TrigramCollocationFinder.from_words(words_corpus)

    print("Showing first",2000,"top-freqent words in the corpus")
    grams = pd.DataFrame(grams) #Building data table to represent selected by POS tagger word features 
    grams.index = range(1,len(grams)+1)
    grams.columns = ["Word", "Frequency"]
    print(grams)

    bi_filter=7
    print("Showing bigrams in the corpus found by Pointwise Mutual Information method")
    print("Applying frequency filter: a bigramm occurs more than",bi_filter,"times")
    bigramFinder.apply_freq_filter(bi_filter)
    bigramPMITable = pd.DataFrame(list(bigramFinder.score_ngrams(bigrams.pmi)), columns=['bigram','PMI']).sort_values(by='PMI', ascending=False)
    bigramPMITable["bigram"]=bigramPMITable["bigram"].apply(lambda x: ' '.join(x))
    print(bigramPMITable)


    tri_filter=5
    print("Showing trigrams in the corpus found by Pointwise Mutual Information method")
    print("Applying frequency filter: a trigramm occurs more than",tri_filter,"times")
    trigramFinder.apply_freq_filter(tri_filter)
    trigramPMITable = pd.DataFrame(list(trigramFinder.score_ngrams(trigrams.pmi)), columns=['trigram','PMI']).sort_values(by='PMI', ascending=False)
    trigramPMITable["trigram"]=trigramPMITable["trigram"].apply(lambda x: ' '.join(x))
    print(trigramPMITable)


    gram_dict=grams.set_index('Word').T.to_dict('list')
    bigramPMIDict=bigramPMITable.set_index('bigram').T.to_dict('list')
    trigramPMIDict=trigramPMITable.set_index('trigram').T.to_dict('list')

    keyword_processor = KeywordProcessor()
    textrank_keyword_processor = KeywordProcessor()

    gram_dict.update(bigramPMIDict)
    bigramPMIDict.update(trigramPMIDict)

#     print(gram_dict)
    print("Extracting keywords from texts using Pointwise Mutual Information method and TextRank")
    text_rank_key_words=dict()
    for i in range(0, len(table_to_process)):
        sentences=table_to_process.loc[i,"pr_abstract"]
        if sentences!=None:
            keywords=get_keywords_by_textrank(sentences)
            if keywords!=None:
                text_rank_key_words.update(keywords)
                print("Text",i,"- Done")

    for keyword in gram_dict.keys():
        parts=keyword.split()
        parts="_".join(parts)
        keyword_processor.add_keyword(keyword,parts)

    for keyword in text_rank_key_words.keys():
        parts=keyword.split()
        parts="_".join(parts)
        textrank_keyword_processor.add_keyword(keyword,parts)

    print("Keywords amount gathered using PMI method")
    print(len(keyword_processor.get_all_keywords()))
    print("Keywords amount gathered using TextRank method")
    print(len(textrank_keyword_processor.get_all_keywords()))

    table_to_process["pr_abstract"]=table_to_process["pr_abstract"].apply(lambda x: merge_two_keywords_methods(x, textrank_keyword_processor, keyword_processor))     

    for i in range(0, len(table_to_process)):
        metadata_table.loc[i,"pr_title"] = table_to_process.loc[i,"pr_title"]
        metadata_table.loc[i,"pr_abstract"] = table_to_process.loc[i,"pr_abstract"]

    print("Comparison of Text Data after Keywords Extraction using Pointwise Mutual Information method and TextRank")
    print(metadata_table[["title","pr_title","abstract","pr_abstract"]])

    print("Extracting keywords from texts using TF/IDF")
    dataset = []
    for i in range(0, len(table_to_process["pr_abstract"])):
        sentences = table_to_process.loc[i,"pr_abstract"]
        if sentences!=None:
            sentences=" ".join(sentences)
            dataset.append(sentences)

    tfIdfVectorizer=TfidfVectorizer(use_idf=True)
    tfIdf = tfIdfVectorizer.fit_transform(dataset)

    index=0
    for i in range(0,len(metadata_table)):
        if table_to_process.loc[i,"pr_abstract"]==None:
            continue
        metadata_table.loc[i,"pr_abstract"]=retain_best_tf_idf_keywords(table_to_process.loc[i,"pr_abstract"], index,tfIdf,tfIdfVectorizer)
        index+=1
    return metadata_table


# ## Stemming or Morphological Analysis (Lemmatisation) 

# In[15]:


def lemmatise_text(sentences):
    if sentences==None:
        return None
    lemmatizer = WordNetLemmatizer()
    for i in range(0, len(sentences)):
        try:
            if sentences[i] == "":
                continue
            words=sentences[i].split()
            lemmatised_words = [lemmatizer.lemmatize(word) for word in words]
            lemmatised_words = ' '.join(lemmatised_words)
            sentences[i]=lemmatised_words
        except:
            print(sentences)
            print(sentences[i])
            traceback.print_exc()
            break
    return sentences


# ## Indexing

# In[16]:


def index_table(es,metadata_table,es_index="covid"):
    for i in range(0,len(metadata_table)):
        metadata_table.iloc[i].to_json(es_index+'.json')
        f = open(es_index+'.json')
        docket_content = f.read()
        row=json.loads(docket_content)
        try:
            es.index(index=es_index, id=i, body=row)
        except:
            traceback.print_exc() 
            print("Error:", "row #"+str(i))


# ## Searching

# In[17]:


def search(es, es_index="covid",
    query={
          "query": {
            "match_phrase":{"publish_time":"2000-08-15"}
          }
        }):
    res = es.search(index=es_index, body=query)
    documents=[]
    for i in range(0, len(res['hits']['hits'])):
        doc=res['hits']['hits'][i]['_source']
        documents.append(doc)
    return documents


# ## Running of the program

# In[18]:


def run_program(path = 'metadata.csv',es_index="covid", query={
              "query": {
                "match_phrase":{"publish_time":"2000-08-15"}
              }
            }):
    #Data loading
    print("Data loading")
    metadata_table= load_data(path)
    print("pr_title and pr_abstract columns have been added")
    print(metadata_table)

    #Indexing
    print("Creating index -",es_index)
    es=create_index(es_index)
    print("Indexing")
    index_table(es,metadata_table,es_index=es_index)
    print("Data indexed.")

    #Sentence splitting, text tokenisation and normalisation
    print("Sentence splitting, text tokenisation and normalisation")
    metadata_table=normalize_table(metadata_table)
    print("Comparison of Text Data after Sentence splitting, text tokenisation and normalisation step")
    print(metadata_table[["title","pr_title","abstract","pr_abstract"]])


    #Selecting keywords
    print("Selecting keywords")
    metadata_table=select_best_keywords(metadata_table)
    print("Comparison of Text Data after Selecting keywords step")
    print(metadata_table[["title","pr_title","abstract","pr_abstract"]])
    
    #Text lemmatisation
    print("Text lemmatisation")
    metadata_table["pr_abstract"]=metadata_table["pr_abstract"].apply(lambda x: lemmatise_text(x))
    metadata_table["pr_title"]=metadata_table["pr_title"].apply(lambda x: lemmatise_text(x))
    print("Comparison of Text Data after Applied Lemmatisation")
    print(metadata_table[["title","pr_title","abstract","pr_abstract"]])

    #Indexing
    print("Creating index -",es_index)
    es=create_index(es_index)
    print("Indexing")
    index_table(es,metadata_table,es_index=es_index)
    
    #Searching in ElasticSearch
    print("Searching in ElasticSearch")
    documents=search(es, es_index=es_index, query=query)
    print("Retrieved documents:")
    pprint.pprint(documents)
    
    return documents,es


# In[19]:


documents,es=run_program(path = 'metadata.csv',es_index="covid", query={
              "query": {
                "match_phrase":{"publish_time":"2000-08-15"}
              }
            })
print("Retrieved documents:")
pprint.pprint(documents)


# In[20]:


documents=search(es, es_index="covid",
    query={
          "query": {
            "match_phrase":{"publish_time":"2000-08-15"}
          }
        })
print("Retrieved documents:")
pprint.pprint(documents)


# In[21]:


documents=search(es, es_index="covid",
    query={
          "query": {
            "match_phrase":{"pr_abstract":"biological diversity"}
          }
        })
print("Retrieved documents:")
pprint.pprint(documents)


# In[22]:


documents=search(es, es_index="covid",
    query={
          "query": {
            "match_phrase":{"abstract":"biological diversity"}
          }
        })
print("Retrieved documents:")
pprint.pprint(documents)


# In[23]:


documents=search(es, es_index="covid",
    query={
          "query": {
            "match_all":{}
          }
        })
print("Retrieved documents:")
pprint.pprint(documents)


# In[ ]:




