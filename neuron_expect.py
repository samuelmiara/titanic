import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
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


columns_after_encoding = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = keras.Sequential([
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)


model.save("titanic_model.keras")


y_pred = (model.predict(X_test) > 0.5).astype("int32").flatten()


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')


new_data = pd.read_csv("test.csv")
new_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
new_data['Sex'] = new_data['Sex'].map({'male': 1, 'female': 0})
new_data.fillna(0, inplace=True)
new_data[['Fare']] = scaler.transform(new_data[['Fare']])
new_data = pd.get_dummies(new_data, columns=['Embarked'])


diff_cols = set(columns_after_encoding) - set(new_data.columns)
for col in diff_cols:
    new_data[col] = 0


new_data = new_data[columns_after_encoding]


model = load_model("titanic_model.keras")
predictions = (model.predict(new_data) > 0.5).astype("int32").flatten()


new_data['Predicted_Survived'] = predictions
new_data.to_csv("predictions_new.csv", index=False)
print("predictions_new.csv")