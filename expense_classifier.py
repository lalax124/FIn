"""
expense_classifier.py  —  Artha AI
────────────────────────────────────
Expense Category Auto-Classifier.

Architecture:
  TF-IDF (word 1-3 gram + char 2-6 gram) → CalibratedLinearSVC
  Trained on 345 hand-labelled + 2525 augmented = 2870 samples
  11 categories, Indian + global transaction descriptions

Cross-validation results (Stratified 5-Fold, held-out folds):
  F1 Macro     : 0.9687  (96.87%)
  F1 Weighted  : 0.9697  (96.97%)
  Accuracy     : 0.9697  (96.97%)

Usage:
  python expense_classifier.py        → train, evaluate, save model
  from expense_classifier import predict_category
"""

import os, re, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.pipeline            import Pipeline, FeatureUnion
from sklearn.svm                 import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection     import StratifiedKFold, cross_validate
from sklearn.metrics             import classification_report, f1_score
from sklearn.calibration         import CalibratedClassifierCV

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artha_classifier.pkl")
DATA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_data.csv")


def preprocess(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text)


def augment(df: pd.DataFrame) -> pd.DataFrame:
    """Word-drop + word-reverse + noise-word augmentation."""
    rows = []
    for _, row in df.iterrows():
        desc, cat = row["description"], row["category"]
        words = preprocess(desc).split()
        rows.append((desc, cat))
        if len(words) > 1:
            rows.append((" ".join(words[:-1]), cat))
            rows.append((" ".join(words[1:]),  cat))
            rows.append((" ".join(words[::-1]),cat))
        for noise in ["payment", "purchase", "charge", "bill", "fee"]:
            rows.append((preprocess(desc) + " " + noise, cat))
    out = pd.DataFrame(rows, columns=["description","category"]).drop_duplicates()
    out["description"] = out["description"].apply(preprocess)
    return out


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word",    ngram_range=(1,3), max_features=30000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,6), max_features=50000, sublinear_tf=True)),
    ])
    clf = CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=3000), cv=3)
    return Pipeline([("features", features), ("clf", clf)])


def train_and_evaluate(data_path=DATA_PATH, save_path=MODEL_PATH):
    df  = pd.read_csv(data_path)
    aug = augment(df)
    X, y = aug["description"].values, aug["category"].values
    classes = sorted(aug["category"].unique())

    print("╔══════════════════════════════════════════════════════════╗")
    print("║        ARTHA AI — Expense Classifier  Training          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Base samples : {len(df):<6}  After augmentation : {len(aug):<6}     ║")
    print(f"║  Categories   : {len(classes)}                                      ║")
    print("╠══════════════════════════════════════════════════════════╣")

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    res = cross_validate(pipeline, X, y, cv=cv,
                         scoring=["f1_macro","f1_weighted","accuracy"],
                         return_train_score=True)

    f1m = res["test_f1_macro"].mean()
    f1w = res["test_f1_weighted"].mean()
    acc = res["test_accuracy"].mean()

    print(f"║  Metric           Train       Val (mean)   Val (±std)   ║")
    print(f"║  {'─'*53}  ║")
    for label, tk, vk in [
        ("Accuracy",    "train_accuracy",    "test_accuracy"),
        ("F1 Macro",    "train_f1_macro",    "test_f1_macro"),
        ("F1 Weighted", "train_f1_weighted", "test_f1_weighted"),
    ]:
        tr = res[tk].mean(); vl = res[vk].mean(); sd = res[vk].std()
        print(f"║  {label:<14}   {tr:.4f}      {vl:.4f}       ±{sd:.4f}    ║")

    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║    F1 Score (Macro)    : {f1m:.4f}  ({f1m*100:.2f}%)              ║")
    print(f"║    F1 Score (Weighted) : {f1w:.4f}  ({f1w*100:.2f}%)              ║")
    print(f"║    Accuracy            : {acc:.4f}  ({acc*100:.2f}%)              ║")
    print("╚══════════════════════════════════════════════════════════╝")

    pipeline.fit(X, y)
    print("\nPer-class report (full augmented fit):")
    print(classification_report(y, pipeline.predict(X), digits=4))

    with open(save_path,"wb") as f:
        pickle.dump({"pipeline": pipeline, "classes": classes}, f)
    print(f"Model saved → {save_path}")

    return {"f1_macro": round(f1m,4), "f1_weighted": round(f1w,4),
            "accuracy": round(acc,4), "n_samples": len(aug)}


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}. Run train_and_evaluate() first.")
    with open(path,"rb") as f:
        return pickle.load(f)


def predict_category(description: str, model_data=None) -> dict:
    """
    Predict expense category for a transaction description.
    Returns: {"category": str, "confidence": float, "top3": list}
    """
    if model_data is None:
        model_data = load_model()
    pipeline = model_data["pipeline"]
    clean    = preprocess(description)
    pred     = pipeline.predict([clean])[0]
    proba    = pipeline.predict_proba([clean])[0]
    top3     = sorted(zip(pipeline.classes_, proba), key=lambda x: -x[1])[:3]
    top3     = [(c, round(float(p),4)) for c,p in top3]
    return {"category": pred, "confidence": round(float(max(proba)),4), "top3": top3}


if __name__ == "__main__":
    train_and_evaluate()

    model = load_model()
    tests = [
        "Zomato biryani order","Airtel postpaid bill","PVR movie 2 tickets",
        "BigBasket weekly groceries","Ola cab airport drop","Apollo pharmacy medicine",
        "monthly rent paid","SIP mutual fund HDFC","Udemy Python ML course",
        "petrol Rs 500","gym annual membership","IRCTC Rajdhani booking",
        "LIC term insurance","Myntra kurta purchase","BESCOM electricity bill",
    ]
    print(f"\n{'Description':<35} {'Prediction':<28} {'Conf':>5}  Top-3")
    print("─"*100)
    for t in tests:
        r = predict_category(t, model)
        top3_str = " | ".join([f"{c} {p:.0%}" for c,p in r["top3"]])
        print(f"{t:<35} {r['category']:<28} {r['confidence']:.0%}    {top3_str}")
