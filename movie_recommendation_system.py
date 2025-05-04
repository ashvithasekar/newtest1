import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Simulated training dataset
X_train = np.array([
    [80, 70, 22, 25], [130, 85, 31, 45], [115, 78, 28, 38], [145, 90, 34, 50],
    [90, 68, 24, 32], [125, 88, 30, 42], [85, 74, 26, 29], [155, 95, 36, 55]
])
y_train = np.array([0, 1, 0, 1, 0, 1, 0, 1])

features = ['Glucose', 'BloodPressure', 'BMI', 'Age']
df_train = pd.DataFrame(X_train, columns=features)
df_train['Diabetic'] = y_train

# 2. Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 3. Take user input
print("\n🩺 Welcome to AI-Powered Diabetes Risk Predictor!")
glucose = int(input("📍 Enter your glucose level (mg/dL): "))
bp = int(input("📍 Enter your blood pressure (mmHg): "))
bmi = float(input("📍 Enter your BMI: "))
age = int(input("📍 Enter your age: "))

# 4. Predict
user_data = np.array([[glucose, bp, bmi, age]])
prediction = model.predict(user_data)[0]
probability = model.predict_proba(user_data)[0]

# 5. Text feedback
print("\n🧾 Health Report:")
if glucose < 100:
    print(f"✔️ Glucose: {glucose} mg/dL — Normal")
elif glucose < 126:
    print(f"⚠️ Glucose: {glucose} mg/dL — Pre-Diabetic")
else:
    print(f"❗ Glucose: {glucose} mg/dL — High (Diabetic)")

if bmi < 18.5:
    bmi_status = "Underweight"
elif bmi < 25:
    bmi_status = "Healthy"
elif bmi < 30:
    bmi_status = "Overweight"
else:
    bmi_status = "Obese"
print(f"📊 BMI: {bmi} — {bmi_status}")

print(f"\n🔍 Final Prediction: {'🚨 Diabetic' if prediction else '✅ Not Diabetic'}")
print(f"🧠 Model Confidence: Not Diabetic = {probability[0]*100:.2f}%, Diabetic = {probability[1]*100:.2f}%")

# 6. Visualizations

# A. Bar Chart - Prediction Probabilities
plt.figure(figsize=(6, 4))
plt.bar(['Not Diabetic', 'Diabetic'], probability, color=['green', 'red'])
plt.title("Bar Chart: Prediction Confidence")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.grid(True)
plt.show()

output:
🩺 Welcome to AI-Powered Diabetes Risk Predictor!
📍 Enter your glucose level (mg/dL): 145
📍 Enter your blood pressure (mmHg): 130
📍 Enter your BMI: 28.5
📍 Enter your age: 45

🧾 Health Report:
❗ Glucose: 145 mg/dL — High (Diabetic)
📊 BMI: 28.5 — Overweight

🔍 Final Prediction: 🚨 Diabetic
🧠 Model Confidence: Not Diabetic = 19.00%, Diabetic = 81.00%
