# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

NUM_CLASSES = 2
from bs4 import BeautifulSoup
import logging
import json
import os
import random


from transformers import DataProcessor, InputExample

import numpy as np
logger = logging.getLogger(__name__)

class AppReviewPretrainProcessor(DataProcessor):
    """Modified from Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        with open(os.path.join(data_dir, "app_reviews_unlabel.json")) as fin:
            reviews = json.load(fin)
        examples = []
        i = 0 
        reviews = [review for review in reviews if review['rating'] <= 3]
        for review in reviews:
            i += 1
            guid = "%s-%s" % ("train", i)
            examples.append(InputExample(guid=guid, text_a=review['content'], text_b=review['title'], label="Y"))
        while len(examples) < 2 * len(reviews):
            r1 = random.randrange(len(reviews))
            r2 = random.randrange(len(reviews))
            if r1 == r2:
                continue
            i += 1
            guid = "%s-%s" % ("train", i)
            examples.append(InputExample(guid=guid, text_a=reviews[r1]['content'], text_b=reviews[r2]['title'], label="N"))
        random.shuffle(examples)
        return examples

    def get_test_examples(self, data_dir):
        #   Only for pretraining. No test data is really required. 
        return self.get_train_examples(data_dir)[:1000]

    def get_labels(self):
        """See base class."""
        return ["N", "Y"]


class AppReviewProcessor(DataProcessor):
    """Modified from Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""  
        ########################################### books##########################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/books/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        positive_reviews = positive_reviews.findAll('review_text')   
        pos=[]
        for i in positive_reviews:
            pos.extend(i)  
        name=[]
        for i in product_name:
            name.extend(i)
        examples1 = []  
        c = list(zip(pos, name))
        random.Random(42).shuffle(c)
      
        pos, name = zip(*c)
        print('books',len(pos),len(name))
        g=0
        for index ,i in enumerate(pos[:800]):
            g+=1        
            guid = "%s-%s" % ("train", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples1.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/books/negative.review').read(), 'lxml')
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        for i in product_name:
            name.extend(i)
        c = list(zip(neg, name))
        random.Random(42).shuffle(c)
        neg, name = zip(*c)
        print('books',len(neg),len(name))
        for index ,i in enumerate(neg[:800]):
            g+=1        
            guid = "%s-%s" % ("train", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples1.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        print(examples1[0])
        print('trainrtraintraintraintraintraintraintrain')
        print(len(examples1))
        ####################dvd#################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/dvd/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        positive_reviews = positive_reviews.findAll('review_text')   
        pos=[]
        
        for i in positive_reviews:
            pos.extend(i)  
        name=[]
        for i in product_name:
            name.extend(i)
        examples2 = []
        c = list(zip(pos, name))
        random.Random(42).shuffle(c)
        pos, name = zip(*c)  
        print('dvd',len(pos),len(name))
        for index ,i in enumerate(pos[:800]):        
            g+=1
            guid = "%s-%s" % ("train", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples2.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/dvd/negative.review').read(), 'lxml')
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        for i in product_name:
            name.extend(i)
        c = list(zip(neg, name))
        random.Random(42).shuffle(c)
        neg, name = zip(*c)
        print('dvd',len(neg),len(name))
        for index ,i in enumerate(neg[:800]):    
            g+=1    
            guid = "%s-%s" % ("train", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples2.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        print(examples2[0])
        print('trainrtraintraintraintraintraintraintrain')
        print(len(examples2))
        ############################electronics#######################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/electronics/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        positive_reviews = positive_reviews.findAll('review_text')   
        pos=[]
        for i in positive_reviews:
            pos.extend(i)  
        name=[]
        for i in product_name:
            name.extend(i)
        examples3 = []
        c = list(zip(pos, name))
        random.Random(42).shuffle(c)
        pos, name = zip(*c)  
        print('electronics',len(pos),len(name))
        for index ,i in enumerate(pos[:800]):        
            g+=1
            guid = "%s-%s" % ("train",g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples3.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/electronics/negative.review').read(), 'lxml')
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        for i in product_name:
            name.extend(i)
        c = list(zip(neg, name))
        random.Random(42).shuffle(c)
        neg, name = zip(*c)
        print('electronics',len(neg),len(name))
        for index ,i in enumerate(neg[:800]):        
            g+=1
            guid = "%s-%s" % ("train", index)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples3.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        print(examples3[0])
        print('trainrtraintraintraintraintraintraintrain')
        print(len(examples3))
        ###################################kitchen##########################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/kitchen_&_housewares/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        positive_reviews = positive_reviews.findAll('review_text')   
        pos=[]
        for i in positive_reviews:
            pos.extend(i)  
        name=[]
        for i in product_name:
            name.extend(i)
        examples4 = []
        c = list(zip(pos, name))
        random.Random(42).shuffle(c)
        pos, name = zip(*c)
        print('kitchen',len(pos),len(name))
        for index ,i in enumerate(pos[:800]):   
            g+=1     
            guid = "%s-%s" % ("train", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples4.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/kitchen_&_housewares/negative.review').read(), 'lxml')
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        for i in product_name:
            name.extend(i)
        c = list(zip(neg, name))
        random.Random(42).shuffle(c)
        neg, name = zip(*c)
        print('kitchen',len(neg),len(name))
        for index ,i in enumerate(neg[:800]):
            g+=1        
            guid = "%s-%s" % ("train", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples4.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        print(examples4[0])
        print('trainrtraintraintraintraintraintraintrain')
        print(len(examples4))
        #books,dvd,electronics,kitchen
        examples=examples2+examples3+examples4
        random.Random(42).shuffle(examples)
        random.Random(42).shuffle(examples1)
        random.Random(42).shuffle(examples2)
        random.Random(42).shuffle(examples3)
        random.Random(42).shuffle(examples4)
               
        #return examples
        #return examples1
        return examples1,examples2,examples3,examples4
    def get_test_examples(self, data_dir):
        """See base class."""
        
        
        ########################################### books##########################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/books/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        
        positive_reviews = positive_reviews.findAll('review_text')   
        pos=[]
        
        for i in positive_reviews:
            pos.extend(i)  
        name=[]
        for i in product_name:
            name.extend(i)
        examples1 = []  
        c = list(zip(pos, name))
        random.Random(42).shuffle(c)
        pos, name = zip(*c)
        g=0
        for index ,i in enumerate(pos[800:]):
            g+=1        
            guid = "%s-%s" % ("test", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples1.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/books/negative.review').read(), 'lxml')
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        for i in product_name:
            name.extend(i)
        c = list(zip(neg, name))
        random.Random(42).shuffle(c)
        neg, name = zip(*c)
        for index ,i in enumerate(neg[800:]):
            g+=1        
            guid = "%s-%s" % ("test", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples1.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        print(examples1[0])
        print('trainrtraintraintraintraintraintraintrain')
        print(len(examples1))
        ####################dvd#################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/dvd/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        positive_reviews = positive_reviews.findAll('review_text')   
        pos=[]
        
        for i in positive_reviews:
            pos.extend(i)  
        name=[]
        for i in product_name:
            name.extend(i)
        examples2 = []
        c = list(zip(pos, name))
        random.Random(42).shuffle(c)
        pos, name = zip(*c)  
        g=0
        for index ,i in enumerate(pos[800:]):        
            g+=1
            guid = "%s-%s" % ("test", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples2.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/dvd/negative.review').read(), 'lxml')
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        for i in product_name:
            name.extend(i)
        c = list(zip(neg, name))
        random.Random(42).shuffle(c)
        neg, name = zip(*c)
        for index ,i in enumerate(neg[800:]):    
            g+=1    
            guid = "%s-%s" % ("test", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples2.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        print(examples2[0])
        print('trainrtraintraintraintraintraintraintrain')
        print(len(examples2))
        ############################electronics#######################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/electronics/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        positive_reviews = positive_reviews.findAll('review_text')   
        pos=[]
        for i in positive_reviews:
            pos.extend(i)  
        name=[]
        for i in product_name:
            name.extend(i)
        examples3 = []
        c = list(zip(pos, name))
        random.Random(42).shuffle(c)
        pos, name = zip(*c)  
        g=0
        for index ,i in enumerate(pos[800:]):        
            g+=1
            guid = "%s-%s" % ("test",g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples3.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/electronics/negative.review').read(), 'lxml')
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        for i in product_name:
            name.extend(i)
        c = list(zip(neg, name))
        random.Random(42).shuffle(c)
        neg, name = zip(*c)
        for index ,i in enumerate(neg[800:]):        
            g+=1
            guid = "%s-%s" % ("test", index)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples3.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        print(examples3[0])
        print('trainrtraintraintraintraintraintraintrain')
        print(len(examples3))
        ###################################kitchen##########################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/kitchen_&_housewares/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        positive_reviews = positive_reviews.findAll('review_text')   
        pos=[]
        for i in positive_reviews:
            pos.extend(i)  
        name=[]
        for i in product_name:
            name.extend(i)
        examples4 = []
        c = list(zip(pos, name))
        random.Random(42).shuffle(c)
        pos, name = zip(*c)
        g=0  
        for index ,i in enumerate(pos[800:]):   
            g+=1     
            guid = "%s-%s" % ("test", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples4.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/kitchen_&_housewares/negative.review').read(), 'lxml')
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        for i in product_name:
            name.extend(i)
        c = list(zip(neg, name))
        random.Random(42).shuffle(c)
        neg, name = zip(*c)
        for index ,i in enumerate(neg[800:]):
            g+=1        
            guid = "%s-%s" % ("test", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples4.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        print(examples4[0])
        print('trainrtraintraintraintraintraintraintrain')
        print(len(examples4))
        #books,dvd,electronics,kitchen
        random.Random(42).shuffle(examples1)
        random.Random(42).shuffle(examples2)
        random.Random(42).shuffle(examples3)
        random.Random(42).shuffle(examples4)
        return examples1,examples2,examples3,examples4
        #return examples4
    def get_labels(self):
        """See base class."""
        return ['positive','negative']


class UnsupAppReviewPretrainProcessor(DataProcessor):
    """Modified from Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        # lines=self._read_tsv(os.path.join(data_dir,"uda.tsv"))      

        # examples = []
        # i = 0 
        
        # for (i,line) in enumerate(lines[:3000]):
        #     i += 1
        #     guid = "%s-%s" % ("unsup", i)
        #     texta,textb=line[0],line[1]
        #     examples.append(InputExample(guid=guid, text_a=texta,text_b=textb,label='負評'))
        
        # print('unsupunsupunsupunsupunsup')
        # print(len(examples))
        # print(examples[0])
        # return examples
        positive_reviews = BeautifulSoup(open('sorted_data_acl/books/unlabeled.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        positive_reviews = positive_reviews.findAll('review_text')   
        pos=[]
        for i in positive_reviews:
            pos.extend(i)  
        name=[]
        for i in product_name:
            x=str(i).replace('<product_name>','')
            x=x.replace('</product_name>','')
            name.append(x)
        print(len(pos))
        print(len(name))
        examples1 = []  
        for index ,i in enumerate(pos):        
            guid = "%s-%s" % ("train", index)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples1.append(InputExample(guid=str(guid), text_a=str(texta),text_b=str(textb), label='positive'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/dvd/unlabeled.review').read(), 'lxml')
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        for i in product_name:
            x=str(i).replace('<product_name>','')
            x=x.replace('</product_name>','')
            name.append(x)
        print(len(neg))
        print(len(name))
        examples2=[]
        for index ,i in enumerate(neg):        
            guid = "%s-%s" % ("train", index)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples2.append(InputExample(guid=guid, text_a=texta,text_b=textb, label='negative'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/electronics/unlabeled.review').read(), 'lxml')
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        for i in product_name:
            x=str(i).replace('<product_name>','')
            x=x.replace('</product_name>','')
            name.append(x)
        print(len(neg))
        print(len(name))
        examples3=[]
        for index ,i in enumerate(neg):        
            guid = "%s-%s" % ("train", index)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples3.append(InputExample(guid=guid, text_a=texta,text_b=textb, label='negative'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/kitchen_&_housewares/unlabeled.review').read(), 'lxml')
        
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        print(len(neg))
        print(len(product_name))
      
        for index,i in enumerate(product_name):
            x=str(i).replace('<product_name>','')
            x=x.replace('</product_name>','')
            name.append(x)
        print(len(name))
     
        examples4=[]
        for index ,i in enumerate(neg):        
            guid = "%s-%s" % ("train", index)
            
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            # print(index)
            # print(textb)
            textb=textb.replace('\n','')            
            # print(textb)
            examples4.append(InputExample(guid=str(guid), text_a=str(texta),text_b=str(textb), label='negative'))
     
        random.Random(42).shuffle(examples1)
        random.Random(42).shuffle(examples2)
        random.Random(42).shuffle(examples3)
        random.Random(42).shuffle(examples4)
        
        return examples1,examples2,examples3,examples4
        #return examples3
class SimilarProcessor(DataProcessor):
    """Modified from Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir,in_out_weight):
        """See base class."""
    
        """See base class."""  
        ############################electronics#######################################
        ###############################ALL target without DVD#######################
        ############################Books#######################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/books/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        book_name=[]
        for i in product_name:
            book_name.extend(i)
        print('---------------------------------------------------------------------------') 
        print('electronic',len(book_name))
        # print(book_name[:1600])
        positive_reviews = BeautifulSoup(open('sorted_data_acl/books/negative.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        for i in product_name:
            book_name.extend(i)
        # print('---------------------------------------------------------------------------') 
        # print('dvd',len(book_name))
        # print(ele_name[:1600])
        random.Random(42).shuffle(book_name)
        ############################Books#######################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/dvd/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        dvd_name=[]
        for i in product_name:
            dvd_name.extend(i)
        print('---------------------------------------------------------------------------') 
        print('electronic',len(dvd_name))
        # print(dvd_name[:1600])
        positive_reviews = BeautifulSoup(open('sorted_data_acl/dvd/negative.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        for i in product_name:
            dvd_name.extend(i)
        # print('---------------------------------------------------------------------------') 
        # print('dvd',len(dvd_name))
        # print(ele_name[:1600])
        random.Random(42).shuffle(dvd_name)
        ############################Electronic#######################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/electronics/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        ele_name=[]
        for i in product_name:
            ele_name.extend(i)
        print('---------------------------------------------------------------------------') 
        print('electronic',len(ele_name))
        # print(ele_name[:1600])
        positive_reviews = BeautifulSoup(open('sorted_data_acl/electronics/negative.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        for i in product_name:
            ele_name.extend(i)
        # print('---------------------------------------------------------------------------') 
        # print('dvd',len(dvd_name))
        # print(ele_name[:1600])
        random.Random(42).shuffle(ele_name)
        ############################kitchen_&_housewares#######################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/kitchen_&_housewares/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        kitchen_name=[]
        for i in product_name:
            kitchen_name.extend(i)
        # print('---------------------------------------------------------------------------') 
        # print('electronic',len(dvd_name))
        # print(dvd_name[:1600])
        positive_reviews = BeautifulSoup(open('sorted_data_acl/kitchen_&_housewares/negative.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        for i in product_name:
            kitchen_name.extend(i)
        # print('---------------------------------------------------------------------------') 
        # print('dvd',len(dvd_name))
        # print(ele_name[:1600])
        random.Random(42).shuffle(kitchen_name)
        all_target=book_name+dvd_name+kitchen_name
        random.Random(42).shuffle(all_target)
        all_target1 = all_target[:int(len(all_target)/2)]
        all_target2 = all_target[int(len(all_target)/2):]
        ############################################################################
        ########################################### book text##########################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/electronics/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        positive_reviews = positive_reviews.findAll('review_text')   
        pos=[]
        for i in positive_reviews:
            pos.extend(i)  
        name=[]
        for i in product_name:
            name.extend(i)
        examples1 = []  
        c = list(zip(pos, name))
        random.Random(42).shuffle(c)
        def rotate(l,n):
            return l[n:]+l[:n]
        pos, name = zip(*c)
        name = rotate(name,10)
        print(type(in_out_weight))
        a = int(in_out_weight*len(pos[:800]))
        #b = int((1.0-in_out_weight)*len(pos[:800]))
        b = 800-a
        print(type(all_target))
        print(type(name))
        print(name[0])
        name = list(name)
        print(name[0])
        negative_sample = all_target[:b] + name[:a]
        random.Random(42).shuffle(negative_sample)
        #print(len(negative_sample))
        
        #print(negative_sample[0])
        #print('books',len(pos),len(name))
        g=0
        for index ,i in enumerate(pos[:800]):
            g+=1        
            guid = "%s-%s" % ("train", g)
            texta , textb =i ,negative_sample[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples1.append(InputExample(guid=g, text_a=texta,text_b=textb, label='irrelevant'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/electronics/negative.review').read(), 'lxml')
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        for i in product_name:
            name.extend(i)
        c = list(zip(neg, name))
        random.Random(42).shuffle(c)
        neg, name = zip(*c)
        print('books',len(neg),len(name))
        for index ,i in enumerate(neg[:800]):
            g+=1        
            guid = "%s-%s" % ("train", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples1.append(InputExample(guid=g, text_a=texta,text_b=textb, label='relevant'))
        print(examples1[0])
        print('trainrtraintraintraintraintraintraintrain')
        print('---------------------------------------------------------------------------') 
        print(len(examples1))
        # ####################dvd#################################
        # positive_reviews = BeautifulSoup(open('sorted_data_acl/dvd/positive.review').read(), 'lxml')
        # product_name = positive_reviews.findAll('product_name')
        # positive_reviews = positive_reviews.findAll('review_text')   
        # pos=[]
        
        # for i in positive_reviews:
        #     pos.extend(i)  
        # name=[]
        # for i in product_name:
        #     name.extend(i)
        # examples2 = []
        # c = list(zip(pos, name))
        # random.Random(42).shuffle(c)
        # pos, name = zip(*c)  
        # print('dvd',len(pos),len(name))
        # for index ,i in enumerate(pos[:800]):        
        #     g+=1
        #     guid = "%s-%s" % ("train", g)
        #     texta,textb=i,name[index]
        #     texta=texta.replace('\n','')
        #     textb=textb.replace('\n','')            
        #     examples2.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        # negative_reviews = BeautifulSoup(open('sorted_data_acl/dvd/negative.review').read(), 'lxml')
        # product_name = negative_reviews.findAll('product_name')
        # negative_reviews = negative_reviews.findAll('review_text')       
        # neg=[]
        # for i in negative_reviews:
        #     neg.extend(i)
        # # print(pos)
        # name=[]
        # for i in product_name:
        #     name.extend(i)
        # c = list(zip(neg, name))
        # random.Random(42).shuffle(c)
        # neg, name = zip(*c)
        # print('dvd',len(neg),len(name))
        # for index ,i in enumerate(neg[:800]):    
        #     g+=1    
        #     guid = "%s-%s" % ("train", g)
        #     texta,textb=i,name[index]
        #     texta=texta.replace('\n','')
        #     textb=textb.replace('\n','')            
        #     examples2.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        # print(examples2[0])
        # print('trainrtraintraintraintraintraintraintrain')
        # print(len(examples2))
        
        
        # examples3 = []
        # c = list(zip(pos, name))
        # random.Random(42).shuffle(c)
        # pos, name = zip(*c)  
        # # print('electronics',len(pos),len(name))
        # for index ,i in enumerate(pos[:800]):        
        #     g+=1
        #     guid = "%s-%s" % ("train",g)
        #     texta,textb=i,name[index]
        #     texta=texta.replace('\n','')
        #     textb=textb.replace('\n','')            
        #     examples3.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        # negative_reviews = BeautifulSoup(open('sorted_data_acl/electronics/negative.review').read(), 'lxml')
        # product_name = negative_reviews.findAll('product_name')
        # negative_reviews = negative_reviews.findAll('review_text')       
        # neg=[]
        # for i in negative_reviews:
        #     neg.extend(i)
        # # print(pos)
        # name=[]
        # for i in product_name:
        #     name.extend(i)
        # c = list(zip(neg, name))
        # random.Random(42).shuffle(c)
        # neg, name = zip(*c)
        # # print('electronics',len(neg),len(name))
        # for index ,i in enumerate(neg[:800]):        
        #     g+=1
        #     guid = "%s-%s" % ("train", index)
        #     texta,textb=i,name[index]
        #     texta=texta.replace('\n','')
        #     textb=textb.replace('\n','')            
        #     examples3.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        # print(examples3[0])
        # print('trainrtraintraintraintraintraintraintrain')
        # print(len(examples3))
        ###################################kitchen##########################################
        # positive_reviews = BeautifulSoup(open('sorted_data_acl/kitchen_&_housewares/positive.review').read(), 'lxml')
        # product_name = positive_reviews.findAll('product_name')
        # positive_reviews = positive_reviews.findAll('review_text')   
        # pos=[]
        # for i in positive_reviews:
        #     pos.extend(i)  
        # name=[]
        # for i in product_name:
        #     name.extend(i)
        # examples4 = []
        # c = list(zip(pos, name))
        # random.Random(42).shuffle(c)
        # pos, name = zip(*c)
        # print('electronics',len(ele_name))
        # print('kitchen',len(pos),len(name))
        # g=0
        # for index ,i in enumerate(pos[:800]):   
        #     g+=1     
        #     guid = "%s-%s" % ("train", g)
        #     texta,textb=i,name[index]
        #     texta=texta.replace('\n','')
        #     textb=textb.replace('\n','')            
        #     examples4.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        # negative_reviews = BeautifulSoup(open('sorted_data_acl/kitchen_&_housewares/negative.review').read(), 'lxml')
        # product_name = negative_reviews.findAll('product_name')
        # negative_reviews = negative_reviews.findAll('review_text')       
        # neg=[]
        # for i in negative_reviews:
        #     neg.extend(i)
        # # print(pos)
        # name=[]
        # for i in product_name:
        #     name.extend(i)
        # c = list(zip(neg, name))
        # random.Random(42).shuffle(c)
        # neg, name = zip(*c)
        # print('kitchen',len(neg),len(name))
        # for index ,i in enumerate(neg[:800]):
        #     g+=1        
        #     guid = "%s-%s" % ("train", g)
        #     texta,textb=i,name[index]
        #     texta=texta.replace('\n','')
        #     textb=textb.replace('\n','')            
        #     examples4.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        # print(examples4[0])
        # print('trainrtraintraintraintraintraintraintrain')
        # print(len(examples4))
        #books,dvd,electronics,kitchen
        # examples=examples2+examples3+examples4
        # random.Random(42).shuffle(examples)
        random.Random(42).shuffle(examples1)
        # random.Random(42).shuffle(examples2)
        # random.Random(42).shuffle(examples3)
        # random.Random(42).shuffle(examples4)
               
        #return examples
        #return examples1
        return examples1
    def get_test_examples(self, data_dir):
        """See base class."""
    
        """See base class."""  
        ########################################### books##########################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/books/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        positive_reviews = positive_reviews.findAll('review_text')   
        pos=[]
        for i in positive_reviews:
            pos.extend(i)  
        name=[]
        for i in product_name:
            name.extend(i)
        examples1 = []  
        c = list(zip(pos, name))
        random.Random(42).shuffle(c)
      
        pos, name = zip(*c)
        print('books',len(pos),len(name))
        g=0
        for index ,i in enumerate(pos[:800]):
            g+=1        
            guid = "%s-%s" % ("train", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples1.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        negative_reviews = BeautifulSoup(open('sorted_data_acl/books/negative.review').read(), 'lxml')
        product_name = negative_reviews.findAll('product_name')
        negative_reviews = negative_reviews.findAll('review_text')       
        neg=[]
        for i in negative_reviews:
            neg.extend(i)
        # print(pos)
        name=[]
        for i in product_name:
            name.extend(i)
        c = list(zip(neg, name))
        random.Random(42).shuffle(c)
        neg, name = zip(*c)
        print('books',len(neg),len(name))
        for index ,i in enumerate(neg[:800]):
            g+=1        
            guid = "%s-%s" % ("train", g)
            texta,textb=i,name[index]
            texta=texta.replace('\n','')
            textb=textb.replace('\n','')            
            examples1.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        print(examples1[0])
        print('trainrtraintraintraintraintraintraintrain')
        print(len(examples1))
        # ####################dvd#################################
        # positive_reviews = BeautifulSoup(open('sorted_data_acl/dvd/positive.review').read(), 'lxml')
        # product_name = positive_reviews.findAll('product_name')
        # positive_reviews = positive_reviews.findAll('review_text')   
        # pos=[]
        
        # for i in positive_reviews:
        #     pos.extend(i)  
        # name=[]
        # for i in product_name:
        #     name.extend(i)
        # examples2 = []
        # c = list(zip(pos, name))
        # random.Random(42).shuffle(c)
        # pos, name = zip(*c)  
        # print('dvd',len(pos),len(name))
        # for index ,i in enumerate(pos[:800]):        
        #     g+=1
        #     guid = "%s-%s" % ("train", g)
        #     texta,textb=i,name[index]
        #     texta=texta.replace('\n','')
        #     textb=textb.replace('\n','')            
        #     examples2.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        # negative_reviews = BeautifulSoup(open('sorted_data_acl/dvd/negative.review').read(), 'lxml')
        # product_name = negative_reviews.findAll('product_name')
        # negative_reviews = negative_reviews.findAll('review_text')       
        # neg=[]
        # for i in negative_reviews:
        #     neg.extend(i)
        # # print(pos)
        # name=[]
        # for i in product_name:
        #     name.extend(i)
        # c = list(zip(neg, name))
        # random.Random(42).shuffle(c)
        # neg, name = zip(*c)
        # print('dvd',len(neg),len(name))
        # for index ,i in enumerate(neg[:800]):    
        #     g+=1    
        #     guid = "%s-%s" % ("train", g)
        #     texta,textb=i,name[index]
        #     texta=texta.replace('\n','')
        #     textb=textb.replace('\n','')            
        #     examples2.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        # print(examples2[0])
        # print('trainrtraintraintraintraintraintraintrain')
        # print(len(examples2))
        ############################electronics#######################################
        positive_reviews = BeautifulSoup(open('sorted_data_acl/electronics/positive.review').read(), 'lxml')
        product_name = positive_reviews.findAll('product_name')
        positive_reviews = positive_reviews.findAll('review_text')   
        pos=[]
        for i in positive_reviews:
            pos.extend(i)  
        ele_name=[]
        for i in product_name:
            ele_name.extend(i)
        print('electronic',len(ele_name))
        # examples3 = []
        # c = list(zip(pos, name))
        # random.Random(42).shuffle(c)
        # pos, name = zip(*c)  
        # # print('electronics',len(pos),len(name))
        # for index ,i in enumerate(pos[:800]):        
        #     g+=1
        #     guid = "%s-%s" % ("train",g)
        #     texta,textb=i,name[index]
        #     texta=texta.replace('\n','')
        #     textb=textb.replace('\n','')            
        #     examples3.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        # negative_reviews = BeautifulSoup(open('sorted_data_acl/electronics/negative.review').read(), 'lxml')
        # product_name = negative_reviews.findAll('product_name')
        # negative_reviews = negative_reviews.findAll('review_text')       
        # neg=[]
        # for i in negative_reviews:
        #     neg.extend(i)
        # # print(pos)
        # name=[]
        # for i in product_name:
        #     name.extend(i)
        # c = list(zip(neg, name))
        # random.Random(42).shuffle(c)
        # neg, name = zip(*c)
        # # print('electronics',len(neg),len(name))
        # for index ,i in enumerate(neg[:800]):        
        #     g+=1
        #     guid = "%s-%s" % ("train", index)
        #     texta,textb=i,name[index]
        #     texta=texta.replace('\n','')
        #     textb=textb.replace('\n','')            
        #     examples3.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        # print(examples3[0])
        # print('trainrtraintraintraintraintraintraintrain')
        # print(len(examples3))
        ###################################kitchen##########################################
        # positive_reviews = BeautifulSoup(open('sorted_data_acl/kitchen_&_housewares/positive.review').read(), 'lxml')
        # product_name = positive_reviews.findAll('product_name')
        # positive_reviews = positive_reviews.findAll('review_text')   
        # pos=[]
        # for i in positive_reviews:
        #     pos.extend(i)  
        # name=[]
        # for i in product_name:
        #     name.extend(i)
        # examples4 = []
        # c = list(zip(pos, name))
        # random.Random(42).shuffle(c)
        # pos, name = zip(*c)
        # print('electronics',len(ele_name))
        # print('kitchen',len(pos),len(name))
        # g=0
        # for index ,i in enumerate(pos[:800]):   
        #     g+=1     
        #     guid = "%s-%s" % ("train", g)
        #     texta,textb=i,name[index]
        #     texta=texta.replace('\n','')
        #     textb=textb.replace('\n','')            
        #     examples4.append(InputExample(guid=g, text_a=texta,text_b=textb, label='positive'))
        # negative_reviews = BeautifulSoup(open('sorted_data_acl/kitchen_&_housewares/negative.review').read(), 'lxml')
        # product_name = negative_reviews.findAll('product_name')
        # negative_reviews = negative_reviews.findAll('review_text')       
        # neg=[]
        # for i in negative_reviews:
        #     neg.extend(i)
        # # print(pos)
        # name=[]
        # for i in product_name:
        #     name.extend(i)
        # c = list(zip(neg, name))
        # random.Random(42).shuffle(c)
        # neg, name = zip(*c)
        # print('kitchen',len(neg),len(name))
        # for index ,i in enumerate(neg[:800]):
        #     g+=1        
        #     guid = "%s-%s" % ("train", g)
        #     texta,textb=i,name[index]
        #     texta=texta.replace('\n','')
        #     textb=textb.replace('\n','')            
        #     examples4.append(InputExample(guid=g, text_a=texta,text_b=textb, label='negative'))
        # print(examples4[0])
        # print('trainrtraintraintraintraintraintraintrain')
        # print(len(examples4))
        #books,dvd,electronics,kitchen
        # examples=examples2+examples3+examples4
        # random.Random(42).shuffle(examples)
        random.Random(42).shuffle(examples1)
        # random.Random(42).shuffle(examples2)
        # random.Random(42).shuffle(examples3)
        # random.Random(42).shuffle(examples4)
               
        #return examples
        #return examples1
        return examples1
    def get_labels(self):
        """See base class."""
        return ["relevant","irrelevant"]
    

class AugAppReviewPretrainProcessor(DataProcessor):
    """Modified from Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_train_examples(self, data_dir):
        """See base class."""
        examples1=[]
        g=0
        with open("transback/book.txt", "r") as f_in:
            lines = (line.rstrip() for line in f_in) 
            lines = list(line for line in lines if line)
            positive_reviews = BeautifulSoup(open('sorted_data_acl/books/unlabeled.review').read(), 'lxml')
            product_name = positive_reviews.findAll('product_name')
            name=[]
            for i in product_name:
                x=str(i).replace('<product_name>','')
                x=x.replace('</product_name>','')
                name.append(x)
            for index ,i in enumerate(lines):        
                g+=1
                guid = "%s-%s" % ("aug", g)
                texta,textb=i,name[index]
                texta=texta.replace('\n','')
                textb=textb.replace('\n','')            
                examples1.append(InputExample(guid=guid, text_a=texta,text_b=textb, label='negative'))
            
            print(len(lines))
        with open("transback/dvd_1.txt", "r") as f_in1:
            lines = (line.rstrip() for line in f_in1) 
            lines_1 = list(line for line in lines if line)
        with open("transback/dvd_2.txt", "r") as f_in2:
            lines = (line.rstrip() for line in f_in2) 
            lines_2 = list(line for line in lines if line)
        with open("transback/dvd_3.txt", "r") as f_in3:
            lines = (line.rstrip() for line in f_in3) 
            lines_3 = list(line for line in lines if line)
        with open("transback/dvd_4.txt", "r") as f_in4:
            lines = (line.rstrip() for line in f_in4) 
            lines_4 = list(line for line in lines if line)
            total_lines=lines_1+lines_2+lines_3+lines_4
            positive_reviews = BeautifulSoup(open('sorted_data_acl/dvd/unlabeled.review').read(), 'lxml')
            product_name = positive_reviews.findAll('product_name')
            name=[]
            for i in product_name:
                x=str(i).replace('<product_name>','')
                x=x.replace('</product_name>','')
                name.append(x)
            examples2=[]
            for index ,i in enumerate(total_lines):
                g+=1        
                guid = "%s-%s" % ("aug", g)
                texta,textb=i,name[index]
                texta=texta.replace('\n','')
                textb=textb.replace('\n','')            
                examples2.append(InputExample(guid=guid, text_a=texta,text_b=textb, label='negative'))
        # print(lines)
            print(len(total_lines))
        with open("transback/electronic.txt", "r") as f_in:
            lines = (line.rstrip() for line in f_in) 
            lines = list(line for line in lines if line)
            positive_reviews = BeautifulSoup(open('sorted_data_acl/electronics/unlabeled.review').read(), 'lxml')
            product_name = positive_reviews.findAll('product_name')
            name=[]
            for i in product_name:
                x=str(i).replace('<product_name>','')
                x=x.replace('</product_name>','')
                name.append(x)
            examples3=[]
            for index ,i in enumerate(lines):        
                g+=1
                guid = "%s-%s" % ("aug", g)
                texta,textb=i,name[index]
                texta=texta.replace('\n','')
                textb=textb.replace('\n','')            
                examples3.append(InputExample(guid=guid, text_a=texta,text_b=textb, label='negative'))
        # print(lines)
            print(len(lines))
        with open("transback/kitchen.txt", "r") as f_in:
            lines = (line.rstrip() for line in f_in) 
            lines = list(line for line in lines if line)
            positive_reviews = BeautifulSoup(open('sorted_data_acl/kitchen_&_housewares/unlabeled.review').read(), 'lxml')
            product_name = positive_reviews.findAll('product_name')
            name=[]
            for i in product_name:
                x=str(i).replace('<product_name>','')
                x=x.replace('</product_name>','')
                name.append(x)
            examples4=[]
            for index ,i in enumerate(lines):        
                g+=1
                guid = "%s-%s" % ("aug", g)
                texta,textb=i,name[index]
                texta=texta.replace('\n','')
                textb=textb.replace('\n','')            
                examples4.append(InputExample(guid=guid, text_a=texta,text_b=textb, label='negative'))
        # print(lines)
            print(len(lines))
        # lines=self._read_tsv(os.path.join(data_dir,"aug.tsv"))      

        # examples = []
        # i = 0 
        
        # for (i,line) in enumerate(lines[:3000]):
        #     i += 1
        #     guid = "%s-%s" % ("aug", i)
        #     texta,textb=line[0],line[1]
        #     examples.append(InputExample(guid=guid, text_a=texta,text_b=textb,label='負評'))
        random.Random(42).shuffle(examples1)
        random.Random(42).shuffle(examples2)
        random.Random(42).shuffle(examples3)
        random.Random(42).shuffle(examples4)
        # return examples1,examples2,examples3,examples4
        print('augaugauaguagugaugaugaugaugaugaug')
        # print(len(examples))
        # print(examples[0])
        
        # return examples1,examples2,examples3,examples4
        return examples3

appreview_processors = {
    "pretrain": AppReviewPretrainProcessor,
    "appreview": AppReviewProcessor,
    "unsup":  UnsupAppReviewPretrainProcessor,
    "aug": AugAppReviewPretrainProcessor,
    "sim":SimilarProcessor
}

appreview_output_modes = {
    "pretrain": "classification", 
    "appreview": "classification",
}

appreview_tasks_num_labels = {
    "pretrain": 2, 
    "appreview": NUM_CLASSES,
}
