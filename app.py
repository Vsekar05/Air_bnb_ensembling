import pandas as pd
import dash
import dash_auth
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash,dcc,html,Input,Output
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
USERNAME_PASSWORD_PAIRS = [
    ['nethu', '12345'],['guvi', 'guvi'],['vignesh','vignesh']
]
#app = JupyterDash(__name__)
app = dash.Dash(__name__)
auth = dash_auth.BasicAuth(app,USERNAME_PASSWORD_PAIRS)
server = app.server

Data=pd.read_csv("https://raw.githubusercontent.com/Vsekar05/Datasets/main/Unit_1_Project_Dataset%20(1).csv")
Data.dropna(inplace=True)

Data.columns

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

scores['AdaBoost'] = {
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

scores['Voting'] = {
        'Train': accuracy_score(y_train, voting.predict(X_train)),
        'Test': accuracy_score(y_test, voting.predict(X_test)),
    }

app.layout=html.Div([html.Div([html.H1(children="Air bnb Data Netherlands Ensembling results using different classifier",
                                       style={'textAlign':'center','color':'red','fontSize':40,
                                                                              'backgroundColor':'black'}),
                               html.Br(),html.Br(),html.Br(),html.Br(),html.Br(),html.Br()]),
                     html.Div([dcc.Dropdown(['Bagging Classifier',
                                            'AdaBoost Classifier',
                                             'Gradient Boosting Classifier',
                                             'Voting Classifier',
                                             'Overall Score for all the Classifier'],'Bagging Classifier',id='Classifier',clearable=False,searchable=False,style=dict(width='45%')),
                               html.Div(id='dd-output-container')])
])
@app.callback(
    Output('dd-output-container','children'),
    Input('Classifier','value')
)
def update_value(value):
  if value=='Bagging Classifier':
    k=evaluate(bagging_clf, X_train, X_test, y_train, y_test)
    return k
  
  elif value=='AdaBoost Classifier':
    l=evaluate(ada_boost_clf, X_train, X_test, y_train, y_test)
    return l
  
  elif value=='Gradient Boosting Classifier':
    m=evaluate(grad_boost_clf, X_train, X_test, y_train, y_test)
    return m
  
  elif value=='Voting Classifier':
    n=evaluate(voting, X_train, X_test, y_train, y_test)
    return n
  
  elif value=='Overall Score for all the Classifier':
    scores_df = pd.DataFrame(scores)
    return scores_df

if __name__ == '__main__':
   app.run_server(debug=True)
