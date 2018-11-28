from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import pickle

#cat=['alt.atheism', 'soc.religion.christian']
#twenty_train = fetch_20newsgroups(subset='train', categories=cat, shuffle=True, random_state=42)
path_youtube="train_data.csv"
#with open(path_youtube,'r+',encoding='utf-8',errors='ignore') as f:
#    Data=f.read()
path_test="test_data.csv"

#print(Data)
Train_df = pd.read_csv(path_youtube)
Test_data=pd.read_csv(path_test)
#print(df)

#for c in df:
#    print(c)

Data_labels=Train_df['SentimentText']
#print(Data_labels.values)
#print(df['AUTHOR'].values)
output=Train_df['Sentiment']
#print(output.values)0

Y_test=Test_data['Sentiment']

#string to vector for machine learning transform
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(Data_labels.values)
#vectorizer = CountVectorizer(tokenizer=extract_features_sk,lowercase=self.lowercase,binary=self.is_binary)
output = open('vect.pickle' , 'wb')
pickle.dump(X_train_counts, output)
output.close()
TF_train_object=TfidfTransformer()
X_tf_train=TF_train_object.fit_transform(X_train_counts)
#print(X_tf_train)

clf=MultinomialNB().fit(X_tf_train,output.values.ravel())

filename = 'finalized_model_multinomial.sav'
pickle.dump(clf, open(filename, 'wb'))