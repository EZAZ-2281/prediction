from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

# Load your dataset and preprocessing as before
df = pd.read_csv('dataset.csv')

# Encode the target column 'Suicidal thoughts'
le = LabelEncoder()
df['Suicidal thoughts'] = le.fit_transform(df['Suicidal thoughts'])

# Separate features and target
X = df.drop('Suicidal thoughts', axis=1)
y = df['Suicidal thoughts']

# Encode categorical features if any
categorical_columns = X.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_columns:
    le_col = LabelEncoder()
    X[col] = le_col.fit_transform(X[col])
    label_encoders[col] = le_col

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multiple classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'k-NN': KNeighborsClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

trained_models = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    trained_models[name] = clf

# Neural network model
def build_neural_network(input_shape):
    model = Sequential()
    model.add(Dense(64, input_dim=input_shape, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

nn_model = build_neural_network(X_train.shape[1])
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
trained_models['Neural Network'] = nn_model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form.to_dict()
    
    # Convert input data to appropriate types
    for col in input_data:
        if col in categorical_columns:
            if input_data[col] in label_encoders[col].classes_:
                input_data[col] = label_encoders[col].transform([input_data[col]])[0]
            else:
                input_data[col] = -1  # Assign a default value or handle unseen labels differently
        else:
            input_data[col] = float(input_data[col])
    
    input_df = pd.DataFrame([input_data])
    input_df = scaler.transform(input_df)
    
    predictions = {}
    for model_name, model in trained_models.items():
        if model_name == 'Neural Network':
            prediction = model.predict(input_df)
            prediction = (prediction > 0.5).astype(int)
        else:
            prediction = model.predict(input_df)
        predictions[model_name] = 'Yes' if prediction[0] == 1 else 'No'
    
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
