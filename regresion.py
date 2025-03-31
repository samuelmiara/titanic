import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

filename = "train.csv"
df = pd.read_csv(filename)

df["Sex"] = df["Sex"].map({"male": 1, "female": 0})
df.fillna(0, inplace=True)
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


scaler = StandardScaler()
df[['Fare']] = scaler.fit_transform(df[['Fare']])


features = ['Fare', 'Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch']
X = pd.get_dummies(df[features], columns=['Embarked'])
y = df['Survived']


columns_after_encoding = X.columns


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

param_grid = {
        'C': [1],
        'solver': ['liblinear']
}





model = LogisticRegression(C=0.5, solver='liblinear', max_iter=200)


model.fit(X_train, y_train)

model = LogisticRegression(max_iter=5000, C=1, solver='liblinear')

model.fit(X_train, y_train)


test_data = pd.read_csv("test.csv")
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})
test_data.fillna(0, inplace=True)
test_data[['Fare']] = scaler.transform(test_data[['Fare']])
test_data = pd.get_dummies(test_data, columns=['Embarked'])


diff_cols = set(columns_after_encoding) - set(test_data.columns)
for col in diff_cols:
    test_data[col] = 0

test_data = test_data[columns_after_encoding]


predictions = model.predict(test_data)
test_data['Predicted_Survived'] = predictions

test_data.to_csv("predictions_logreg.csv", index=False)
print("działą")
