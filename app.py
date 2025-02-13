from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and column names
model = joblib.load('decision_tree.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('form.html')  # Render the input form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        user_input = {
            "person_age": int(request.form['person_age']),
            "person_income": float(request.form['person_income']),
            "person_home_ownership": request.form['person_home_ownership'],
            "person_emp_length": int(request.form['person_emp_length']),
            "loan_intent": request.form['loan_intent'],
            "loan_amnt": float(request.form['loan_amnt']),
            "loan_int_rate": float(request.form['loan_int_rate']),
            "loan_status": int(request.form['loan_status']),
            "loan_percent_income": float(request.form['loan_percent_income']),
            "cb_person_default_on_file": request.form['cb_person_default_on_file'],
            "cb_person_cred_hist_length": int(request.form['cb_person_cred_hist_length']),
        }

        # Convert user input to a DataFrame
        input_df = pd.DataFrame([user_input])

        # One-hot encode categorical features
        categorical_features = ['person_home_ownership', 'loan_intent', 'cb_person_default_on_file']
        input_df_encoded = pd.get_dummies(input_df, columns=categorical_features, drop_first=False)

        # Align the input DataFrame with the model's expected columns
        final_input_df = input_df_encoded.reindex(columns=model_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(final_input_df)

        # Map numeric prediction to loan grade
        loan_grade_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
        predicted_grade = loan_grade_mapping.get(int(prediction[0]), "Unknown")

        # Return the prediction
        return render_template('result.html', prediction=predicted_grade)

    except Exception as e:
        # Return error details for debugging
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)