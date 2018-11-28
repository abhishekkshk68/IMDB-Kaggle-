import csv
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

filename = 'finalized_model_multinomial.sav'
loaded_model = pickle.load(open(filename,'rb'))
path_test="test_data.csv"
Test_data=pd.read_csv(path_test)
Y_test=Test_data['Sentiment']
count_vect = CountVectorizer()
Y_count_vector=count_vect.fit_transform(Y_test)
#X_train_counts = count_vect.fit_transform(Data_labels.values)
TF_train_object=TfidfTransformer()
#print (Y_test)
Y_predict=TF_train_object.fit_transform(Y_count_vector)

#loaded_model = pickle.load(open(filename,'rb'))
prediction_result=loaded_model.predict(Y_predict)
print(prediction_result)