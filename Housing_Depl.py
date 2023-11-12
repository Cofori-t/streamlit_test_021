import streamlit as st
st.title("Housing Prices Prediction")

st.write("""
### # reading
import pandas as pd
url="https://drive.google.com/file/d/15J_Xi1_TSl6iwy9M62G8kIk2JprDLx_E/view?usp=sharing"
path = "https://drive.google.com/uc?export=download&id="+url.split("/")[-2]
housing = pd.read_csv(path)
#housing = pd.read_csv('WBS/ML/housing-deployment-reg.csv')

# train test split
from sklearn.model_selection import train_test_split
X = housing.drop(columns="SalePrice")
y = housing["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=8)

# pipeline
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(),
        KNeighborsRegressor())

# parameter grid for pipeline
pipe_params = {
    'simpleimputer__strategy':['median', 'mean'],
    'standardscaler__with_mean':[True, False],
    'kneighborsregressor__n_neighbors': list(range(1, 20)),
    'kneighborsregressor__weights': ['uniform', 'distance'],
    'kneighborsregressor__p': [1, 2],
    'kneighborsregressor__algorithm': ['ball_tree', 'kd_tree', 'brute']}

# grid search
from sklearn.model_selection import GridSearchCV
trained_pipe = GridSearchCV(pipe,
                            pipe_params,
                            cv = 5)
trained_pipe.fit(X_train, y_train)

# test accuracy on the test set
from sklearn.metrics import r2_score

y_pred = trained_pipe.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(r2).

""")


