# Define a function to predict suicidal thought using a specified model
# def predict_suicidal_thought(input_data, model_name='Random Forest'):
#     # Convert input data to a DataFrame
#     input_df = pd.DataFrame([input_data])

#     # Apply the same preprocessing steps
#     for col in categorical_columns:
#         if col in input_df.columns:
#             # Handle unseen labels
#             input_df[col] = input_df[col].apply(lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1)

#     input_df = scaler.transform(input_df)

#     # Make prediction
#     model = trained_models[model_name]
#     prediction = model.predict(input_df)
#     return 'Yes' if prediction[0] == 1 else 'No'

# # Example usage of the prediction function
# input_data = {
#     'Age': 30,  # Provide numerical value instead of 'between 22 to 35 years'
#     'Gender': 'Male',
#     'Education': 'Graduate',
#     'Live with': 'Family',
#     'Conflict with law': 'No',
#     'Most used drugs': 'Alcohol',
#     'Motive about drug': 'Curiosity',
#     'motivation by friends': 'Yes',
#     'Spend most time': 'Work',
#     'Mental/emotional problem': 'Yes',
#     'Family relationship': 'Good',
#     'Financials of family': 'Stable',
#     'Addicted person in family': 'No',
#     'no. of friends': 5,
#     'Withdrawal symptoms': 'No',
#     'Satisfied with workplace': 'Yes',
#     'Case in court': 'No',
#     'Living with drug user': 'No',
#     'Smoking': 'No',
#     'Easy to control use of drug': 'Yes',
#     'Frequency of drug usage': 'Occasionally',
#     'Taken drug while experiencing stress': 'Yes'
# }

# # Predict using each model
# for model_name in classifiers.keys():
#     print(f"{model_name} prediction: {predict_suicidal_thought(input_data, model_name)}")
