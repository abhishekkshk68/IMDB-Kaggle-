from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import jieba

import pandas as pd

#cat=['alt.atheism', 'soc.religion.christian']
#twenty_train = fetch_20newsgroups(subset='train', categories=cat, shuffle=True, random_state=42)
path_youtube="train.csv"
#with open(path_youtube,'r+',encoding='utf-8',errors='ignore') as f:
#    Data=f.read()
path_test="test.csv"

#print(Data)
Train_df = pd.read_csv(path_youtube)
Test_df=pd.read_csv(path_test)
#print(df)

#for c in df:
#    print(c)

Data_labels=Train_df['Ch1_zh']+Train_df['Ch2_zh']
text_list=[]
for x in Data_labels:
    text_list.append(' '.join(jieba.cut(x, HMM=False)))
print(text_list)
#print(Data_labels.values)
#print(df['AUTHOR'].values)
output=Train_df['label']
#print(output.values)0

Y_test=Test_df['Ch1_zh']+Test_df['Ch2_zh']
Y_list=[]
for x in Y_test:
    Y_list.append(' '.join(jieba.cut(x, HMM=False)))
print(Y_list)
#string to vector for machine learning transform
count_vect = CountVectorizer(token_pattern=r'(?u)\b\w+\b', stop_words=None)
X_train_counts = count_vect.fit_transform(text_list)
TF_train_object=TfidfTransformer()
X_tf_train=TF_train_object.fit_transform(X_train_counts)
#print(X_tf_train)

clf=MultinomialNB().fit(X_tf_train,output.values.ravel())

Y_count_vector=count_vect.transform(Y_list)
#print (Y_test)
Y_predict=TF_train_object.transform(Y_count_vector)

prediction_result=clf.predict(Y_predict)
#print(prediction_result)
#print the output
doc_title=Test_df['title1']
c=0
for doc, category in zip(doc_title, prediction_result):
    print('%r => %s' % (doc, category))
    c=c+1
print(c)