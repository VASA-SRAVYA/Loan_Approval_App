from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('loan_model.pkl', 'rb'))

app = Flask(__name__)

# Landing Page
@app.route('/')
def landing():
    return render_template('home.html')  # Welcome screen

# Prediction Form Page
@app.route('/predict_form')
def predict_form():
    return render_template('index.html')  # Loan input form

# Prediction Logic
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [x for x in request.form.values()]
        data = []

        # Manual encoding based on model training
        encoders = {
            "Gender": {"Male": 1, "Female": 0},
            "Married": {"Yes": 1, "No": 0},
            "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
            "Education": {"Graduate": 1, "Not Graduate": 0},
            "Self_Employed": {"Yes": 1, "No": 0},
            "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0}
        }

        # Encode categorical and parse numeric
        data.append(encoders["Gender"][features[0]])
        data.append(encoders["Married"][features[1]])
        data.append(encoders["Dependents"][features[2]])
        data.append(encoders["Education"][features[3]])
        data.append(encoders["Self_Employed"][features[4]])
        data.append(float(features[5]))  # ApplicantIncome
        data.append(float(features[6]))  # CoapplicantIncome
        data.append(float(features[7]))  # LoanAmount
        data.append(float(features[8]))  # Loan_Amount_Term
        data.append(float(features[9]))  # Credit_History
        data.append(encoders["Property_Area"][features[10]])

        prediction = model.predict([np.array(data)])
        output = 'Approved ✅' if prediction[0] == 1 else 'Rejected ❌'

        return render_template('index.html', prediction_text=f'Loan Status: {output}')

    except ValueError as ve:
        return render_template('index.html', prediction_text=f"Value Error: {ve}")

    except KeyError as ke:
        return render_template('index.html', prediction_text=f"Input Error: {ke} is not a valid input.")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Unexpected Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
