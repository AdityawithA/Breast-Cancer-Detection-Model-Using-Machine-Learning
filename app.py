from flask import Flask, render_template, request, send_file
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.datasets import load_breast_cancer
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset
data = load_breast_cancer()
feature_names = data.feature_names


# ================= Feature Importance Chart =================
def generate_feature_importance_chart():
    importances = model.feature_importances_
    indices = importances.argsort()[-10:]

    plt.figure(figsize=(6, 4))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importance")

    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    plt.close()

    return base64.b64encode(image_png).decode('utf-8')


# ================= Home =================
@app.route("/")
def home():
    return render_template("index.html", features=feature_names)


# ================= Manual Prediction =================
@app.route("/predict", methods=["POST"])
def predict():
    input_features = []

    for feature in feature_names:
        value = request.form.get(feature)
        input_features.append(float(value))

    final_input = np.array([input_features])

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0].tolist()

    confidence = round(max(probability) * 100, 2)

    if prediction == 0:
        result = "Malignant Tumor Detected"
    else:
        result = "Benign Tumor Detected"

    chart = generate_feature_importance_chart()

    return render_template("index.html",
                           features=feature_names,
                           prediction_text=result,
                           confidence=confidence,
                           probabilities=probability,
                           chart=chart,
                           malignant=(prediction == 0))


# ================= CSV Upload =================
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]

    if file:
        df = pd.read_csv(file)

        prediction = model.predict(df)[0]
        probability = model.predict_proba(final_input)[0].tolist()

        confidence = round(max(probability) * 100, 2)

        if prediction == 0:
            result = "Malignant Tumor Detected"
        else:
            result = "Benign Tumor Detected"

        chart = generate_feature_importance_chart()

        return render_template("index.html",
                               features=feature_names,
                               prediction_text=result,
                               confidence=confidence,
                               probabilities=probability,
                               chart=chart,
                               malignant=(prediction == 0))

    return render_template("index.html", features=feature_names)


# ================= PDF Download =================
@app.route("/download")
def download_pdf():
    result = request.args.get("result")
    confidence = request.args.get("confidence")

    file_path = "prediction_report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("Breast Cancer Prediction Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Result: " + result, styles["Normal"]))
    elements.append(Paragraph("Confidence: " + confidence + "%", styles["Normal"]))

    doc.build(elements)

    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

