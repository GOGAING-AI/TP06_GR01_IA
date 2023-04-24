import pandas as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import seaborn as sns


#importation de notre dataset "C:\\Users\\FOGUEN LUCAS\Documents\\Pojet IA\\TP06_GR01_AI\\"
dataset=pd.read_csv('C:\\Users\\FOGUEN LUCAS\Documents\\Pojet IA\\TP06_GR01_AI\\Churn-Modelling_Feture_Engineering_Pickle_and_hyperParameterTuning.csv')

print(dataset)
#Affichage du nombre de colone
print(dataset.shape)
#Affichage de la moyenne des colonnes
print(dataset.mean())
#Affichage de la mediane de chaque colonne
print(dataset.median())
#Affichage des colonnes de notre dataset
print(dataset.columns)
#Affichage d'un certain nombre d'information Statistique
print(dataset.describe())
#Afficharge du graphique
dataset.plot()
#Affichage du boxplot des membre actif
dataset.boxplot()
#Affichage d'in pairplot
sns.pairplot(dataset)
#Affichage du paiplot en fonction de l'existance
sns.pairplot(dataset, hue='Exited')

#preparation des donnees a l'entrainnement
X=dataset.iloc[:,:-1] #ici on prend toutes ligne y compris toutes les colonnes sauf celle de la derniere colonne
Y=dataset.iloc[:,-1] #on prend toutes lignes de la derniere colonne
#Division de la dataset por l'entrainnement
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state=42)

#affichage des infos dur l'entrainnement et le test
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
#   Creation de  Model
model=LogisticRegression()
model.fit(x_train, y_train)

#Test de Model
predictions = model.predict(x_test)
print("Pourcentage de test {}%".format(model_score(x_test, y_test)*100))
print(predictions)
print(y_test)

#Evaluation du model
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))

#Saving model
pickle.dump(regressor, open('model.pkl','wb'))
