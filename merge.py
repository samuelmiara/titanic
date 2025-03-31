import pandas as pd


df = pd.read_csv("predictions_logreg.csv")
df1 = pd.read_csv("predictions_new.csv")
df2 = pd.read_csv("predictions_rf.csv")


df.drop(['Fare', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked_0', 'Embarked_C', 'Embarked_Q', 'Embarked_S'],
        axis=1, inplace=True)


df['PassengerId'] = range(892, 892 + len(df))


for i in range(len(df)):
    suma = df.loc[i, 'Predicted_Survived'] + df1.loc[i, 'Predicted_Survived'] + df2.loc[i, 'Predicted_Survived']

    if suma == 0 or suma == 1:
        df.loc[i, 'Predicted_Survived'] = 0
    else:
        df.loc[i, 'Predicted_Survived'] = 1

df['Survived']=df['Predicted_Survived']
df = df[['PassengerId', 'Survived']]


df.to_csv('wyniki.csv', index=False, encoding='utf-8')

print("Plik 'wyniki.csv' został zapisany poprawnie.")
