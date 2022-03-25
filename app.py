import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


Data=pd.read_csv("https://raw.githubusercontent.com/Vsekar05/Datasets/main/Unit_1_Project_Dataset%20(1).csv")
Data.dropna(inplace=True)

feature_columns = ["price" ,"review_scores_rating","review_scores_accuracy","review_scores_cleanliness","review_scores_checkin","review_scores_communication",
                                                     "review_scores_location","review_scores_value","host_response_rate","host_response_rate"]
X = Data[feature_columns]
y = Data.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate(model, X_train, X_test, y_train, y_test):
    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    print("TRAINIG RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_train, y_train_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_train, y_train_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

    print("TESTING RESULTS: \n===============================")
    clf_report = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True))
    print(f"CONFUSION MATRIX:\n{confusion_matrix(y_test, y_test_pred)}")
    print(f"ACCURACY SCORE:\n{accuracy_score(y_test, y_test_pred):.4f}")
    print(f"CLASSIFICATION REPORT:\n{clf_report}")

tree = DecisionTreeClassifier()
bagging_clf = BaggingClassifier(base_estimator=tree, n_estimators=1500, random_state=42)
bagging_clf.fit(X_train, y_train)

scores = {
    'Bagging Classifier': {
        'Train': accuracy_score(y_train, bagging_clf.predict(X_train)),
        'Test': accuracy_score(y_test, bagging_clf.predict(X_test)),
    },
}
ada_boost_clf = AdaBoostClassifier(n_estimators=30)
ada_boost_clf.fit(X_train, y_train)

scores['AdaBoost Classifier'] = {
        'Train': accuracy_score(y_train, ada_boost_clf.predict(X_train)),
        'Test': accuracy_score(y_test, ada_boost_clf.predict(X_test)),
    }

grad_boost_clf = GradientBoostingClassifier(n_estimators=10, random_state=42)
grad_boost_clf.fit(X_train, y_train)

scores['Gradient Boosting'] = {
        'Train': accuracy_score(y_train, grad_boost_clf.predict(X_train)),
        'Test': accuracy_score(y_test, grad_boost_clf.predict(X_test)),
    }

estimators = []
log_reg = LogisticRegression(solver='liblinear')
estimators.append(('Logistic', log_reg))

tree = DecisionTreeClassifier()
estimators.append(('Tree', tree))

svm_clf = SVC(gamma='scale')
estimators.append(('SVM', svm_clf))

voting = VotingClassifier(estimators=estimators)
voting.fit(X_train, y_train)

scores['Voting Classifier'] = {
        'Train': accuracy_score(y_train, voting.predict(X_train)),
        'Test': accuracy_score(y_test, voting.predict(X_test)),
    }
scores_df = pd.DataFrame(scores)


Classifier=st.selectbox(label="Select the Classifiers", options=["Bagging Classifier","AdaBoost Classifier","Gradient Boosting","Voting Classifier","Overall Score","View scores as a bar plot"])

if Classifier=="Bagging Classifier":
  k=evaluate(bagging_clf, X_train, X_test, y_train, y_test)
  st.write('The scores for Bagging Classifier on the given data',k)

elif Classifier=="AdaBoost Classifier":
  l=evaluate(ada_boost_clf, X_train, X_test, y_train, y_test)
  st.write('The scores for AdaBoost Classifier on the given data',l)

elif Classifier=="Gradient Boosting":
  m=evaluate(grad_boost_clf, X_train, X_test, y_train, y_test)
  st.write('The scores for Gradient Boosting on the given data',m)

elif Classifier=="Voting Classifier":
  n=evaluate(voting, X_train, X_test, y_train, y_test)
  st.write('The scores for Voting Classifier on the given data',n)
  
elif Classifier=="Overall Score":
  st.write('The overall scores on the given data',scores_df)

elif Classifier=="View scores as a bar plot":
  st.bar_chart(scores_df,width=15, height=8)
