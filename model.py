import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load and clean Titanic data in the exact feature order used by app.py
df = sns.load_dataset("titanic")[["survived", "pclass", "sex", "age", "fare"]].dropna().copy()
df["sex"] = (df["sex"] == "female").astype(int)

features = ["pclass", "sex", "age", "fare"]
X = df[features]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Save directly into your Streamlit app folder
joblib.dump(model, r"D:\python\Streamlit\model.joblib")
print("Saved model.joblib successfully!")