from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score


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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

param_dist = {
    'n_estimators': [400],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}


rf = RandomForestClassifier()
random_search = RandomizedSearchCV(rf, param_dist, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1)
random_search.fit(X_train, y_train)


print(f'Best parameters: {random_search.best_params_}')


best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

test_data = pd.read_csv("test.csv")
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 0})
test_data.fillna(0, inplace=True)
test_data[['Fare']] = scaler.transform(test_data[['Fare']])
test_data = pd.get_dummies(test_data, columns=['Embarked'])


diff_cols = set(X.columns) - set(test_data.columns)
for col in diff_cols:
    test_data[col] = 0

test_data = test_data[X.columns]


predictions = best_model.predict(test_data)
test_data['Predicted_Survived'] = predictions


test_data.to_csv("predictions_rf.csv", index=False)
print("Predykcje zapisane do predictions_rf.csv")
