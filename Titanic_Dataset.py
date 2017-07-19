import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
%matplotlib inline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

dataset=pd.read_csv('./train.csv')

dataset_columns=dataset.drop(['Survived'],axis=1).columns
class Transform(BaseEstimator,TransformerMixin):


    def __init__(self,select_dict={},default=False,list_of_attributes=dataset_columns):
        self.attributes=select_dict

        for i in list_of_attributes:
            if i not in self.attributes:
                self.attributes[i]=default
        self.func={
            'PassengerId': self._transform_PassengerId,
            'Pclass': self._transform_Pclass,
            'Name': self._transfrom_Name,
            'Sex': self._transform_Sex,
            'Age': self._transform_Age,
            'SibSp': self._transform_SibSp,
            'Parch': self._transfrom_Parch,
            'Ticket': self._transform_Ticket,
            'Fare': self._transform_Fare,
            'Cabin': self._transform_Cabin,
            'Embarked': self._transform_Embarked
        }
        self.Transformed_Data=[]

    def fit(self,X,y=None):
        return self

    def _transform_PassengerId(self,X):
        #Not much to do!!
        return self.Transformed_Data.append(X)

    def _transform_Pclass(self,X):
        return self.Transformed_Data.append(4-X)

    def _transfrom_Name(self,X):
        Extracted_List=[]
        for i in X:
            for j in range(len(i)):
                if i[j]==',':
                    break
            Extracted_List.append(i[:j])
        #first name extraction done
        #Now encoding labels ;) don't want to use LabelEncoder!
        labels={}
        count=0
        for i in range(len(Extracted_List)):
            if Extracted_List[i] not in labels:
                labels[Extracted_List[i]]=count
                count+=1
            Extracted_List[i]=labels[Extracted_List[i]]
        return self.Transformed_Data.append(pd.DataFrame(data=Extracted_List,columns=['Name']))

    def _transform_Sex(self,X):
        Transformed_list=[i for i in map(lambda x: 0 if x=='male' else 1,X)]
        return self.Transformed_Data.append(pd.DataFrame(data=Transformed_list,columns=['Sex']))

    def _transform_Age(self,X):
        #I don't know about Age yet but let's do something
        fill=X.mean()
        Age=X.copy()
        return self.Transformed_Data.append(Age.fillna(fill))

    def _transform_SibSp(self,X):
        #Not much to do
        return self.Transformed_Data.append(X)

    def _transfrom_Parch(self,X):
        #Here as well!! Not much to do
        return self.Transformed_Data.append(X)

    def _transform_Ticket(self,X):
        Ticket_list=[]
        labels={}
        count=0
        for i in X:
            if i not in labels:
                labels[i]=count
                count+=1
            Ticket_list.append(labels[i])

        return self.Transformed_Data.append(pd.DataFrame(data=Ticket_list,columns=['Ticket']))

    def _transform_Fare(self,X):
        Fare_scaled=X.copy()
        Fare_scaled.fillna(Fare_scaled.mean(),inplace=True)
        Fare_scaled=(Fare_scaled-Fare_scaled.mean())/Fare_scaled.std()
        return self.Transformed_Data.append(Fare_scaled)

    def _transform_Cabin(self,X):
        list_Cabin=[i for i in map(lambda x: x[0] if type(x)==str else 'U',X)]
        #Converted to deck now
        #Now converting to integers
        count=0
        labels={}
        for i in range(len(list_Cabin)):
            if list_Cabin[i] not in labels:
                labels[list_Cabin[i]]=count
                count+=1
            list_Cabin[i]=labels[list_Cabin[i]]
        #labels encoded to integers
        return self.Transformed_Data.append(pd.DataFrame(data=list_Cabin,columns=['Cabin']))

    def _transform_Embarked(self,X):
        labels={}
        count=0
        list_Embarked=[]
        for i in range(len(X)):
            if i not in labels:
                labels[X[i]]=count
                count+=1
            list_Embarked.append(labels[X[i]])
        return self.Transformed_Data.append(pd.DataFrame(data=list_Embarked,columns=["Embarked"]))

    def transform(self,X,y=None):
        self.Transformed_Data=[]
        for i in self.attributes:
            if self.attributes[i]:
                self.func[i](X[i])

        return pd.concat(self.Transformed_Data,axis=1)


predict_pipeline=Pipeline([
    ('Transformer',Transform(select_dict={'PassengerId': False},default=True)),
    ('Estimator',for_reg)
])

X_train=dataset.drop('Survived',axis=1)
y_train=dataset["Survived"]
predict_pipeline.fit(X=X_train,y=y_train)
dataset_test=pd.read_csv('./test.csv')
y_test_pred=predict_pipeline.predict(dataset_test)
y_pred_d=pd.DataFrame(y_test_pred,columns=['Survived'])
result=pd.concat([dataset_test['PassengerId'],y_pred_d],axis=1)
