from flask import *  
from IPython.display import display, Image
from PIL import Image
import os

import json
import keras
from time import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
import string
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.models import Model,load_model
from keras.preprocessing import image
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input,Dense,Dropout,Embedding,LSTM
from keras.layers.merge import add
import collections

def readTextFile(path):
    with open(path) as f:
        captions=f.read()
    return captions

# # Data Cleaning
def clean_text(sentence):
    sentence=sentence.lower()
    sentence=re.sub('[^a-z]+',' ',sentence)
    sentence=sentence.split()
    sentence=[s for s in sentence if len(s)>1]
    sentence=' '.join(sentence)
    return sentence

def preprocess_img(img):
    img=image.load_img(img,target_size=(244,244))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)  # batch+ image shape for resnet50 
    
    #Normalization
    img=preprocess_input(img)
    return img

def encode_image(img):
    img=preprocess_img(img)
    feature_vector=model_new.predict(img)
    feature_vector=feature_vector.reshape((-1,))
    return feature_vector

def predict_caption(photo):
    in_text = "startseq"
    
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model1.predict([photo,sequence])
        ypred = ypred.argmax()   # Greedily take.
        word = idx_to_word[ypred]
        in_text+= ' ' +word
        
        if word =='endseq':
            break
        
        
    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption


def solve(path):
    photo=encode_image(path).reshape((1,2048))
    #i = plt.imread(path)
    #plt.imshow(i)
    #plt.axis("off")
    #plt.show()
    caption = predict_caption(photo)
    return caption


###########################################################
##########################################################
###########################################################
###########################################################
##########################################################




app = Flask(__name__)  
 
@app.route('/')  
def upload():  
    return render_template("index.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        f = request.files['img'] 
        var=os.path.join(app.root_path, 'static/'+f.filename)
        f.save(var)  

        print(var)     

        cap = solve(var)
        #cap="Hello World"
        img = Image.open(var)
        #img.show()

        image_path = f.filename


        #print(image_path)

        return render_template("success.html", name = cap, path_to_image = image_path)  
  


##############################################################################
###################################################################
#################################################################

captions=readTextFile('archive/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt')

captions=captions.split('\n')[:-1]

descriptions={}

for x in captions:
    first,second=x.split('\t')
    img_name=first.split('.')[0]
    
    if descriptions.get(img_name) is None:
        descriptions[img_name]=[]
    descriptions[img_name].append(second)

for key,cap_list in descriptions.items():
    for i in range(len(cap_list)):
        cap_list[i]=clean_text(cap_list[i])
    
with open('description.txt','w') as f:
    f.write(str(descriptions))

descriptions=None
with open ('description.txt','r') as f:
    descriptions=f.read()
json_acc=descriptions.replace("'","\"")
descriptions=json.loads(json_acc)

vocab=set()
for key in descriptions.keys():
    [vocab.update(sentence.split()) for sentence in descriptions[key]]

# filter words from vocab according to certain threshold frequency
tot_words=[]
for key in descriptions.keys():
    [tot_words.append(i) for des in descriptions[key] for i in des.split()]

counter=collections.Counter(tot_words)
freq_cnt=dict(counter)

sorted_freq_cnt=sorted(freq_cnt.items(),reverse=True,key=lambda x:x[1])

threshold=10
sorted_freq_cnt=[x for x in sorted_freq_cnt if x[1]>threshold]
total_words=[x[0] for x in sorted_freq_cnt]

train_file_data=readTextFile('archive/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt')
test_file_data=readTextFile('archive/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt')

train=[row.split('.')[0] for row in train_file_data.split('\n')[:-1]]
test=[row.split('.')[0] for row in test_file_data.split('\n')[:-1]]

# Prepare descriptions for training data
#Tweak- Add <s> and <e> token to our training data
train_descriptions={}
for img_id in train:
    train_descriptions[img_id]=[]
    for cap in descriptions[img_id]:
        cap_to_append='startseq '+cap+' endseq'
        train_descriptions[img_id].append(cap_to_append)

model=ResNet50(weights='imagenet',input_shape=(224,224,3))

model_new=Model(model.input,model.layers[-2].output)



idx_to_word={}
word_to_idx = {}
for i,word in enumerate(total_words):
    word_to_idx[word]=i+1
    idx_to_word[i+1]=word

# Two special words:
idx_to_word[1846]='startseq'
idx_to_word[1847]='endseq'
word_to_idx['startseq']=1846
word_to_idx['endseq']=1847

max_len=0
for key in train_descriptions.keys():
    for cap in train_descriptions[key]:
        max_len=max(max_len,len(cap.split()))

model1 = load_model("model_weights/model_weights/model_9.h5")

app.run(debug = True)