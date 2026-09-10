import dill
import pandas as pd
import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/homework.csv')

def filter_data(df):
    df = df.copy()
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long']
    return df.drop(columns_to_drop, axis=1)


def year_outliers_clean(df):
    df = df.copy()
    data_year = df.year
    q25 = data_year.quantile(0.25)
    q75 = data_year.quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
    df.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])

    return df

def short_model(df):
    import pandas as pd
    df = df.copy()
    df['short_model'] = df['model'].apply(lambda x: x.lower().split(' ')[0] if not pd.isna(x) else x)
    return df

def age_category(df):
    df = df.copy()
    df['age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df

numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                        ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                          ('encoder', OneHotEncoder(handle_unknown='ignore'))])

column_transformer = ColumnTransformer(transformers=[
    ('num', numerical_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
    ('cat', categorical_transformer, make_column_selector(dtype_include=object))], remainder='passthrough')

preprocessor = Pipeline(steps=[
    ('filter', FunctionTransformer(filter_data)),
    ('year_outliers', FunctionTransformer(year_outliers_clean)),
    ('add_short_model', FunctionTransformer(short_model)),
    ('age_category', FunctionTransformer(age_category)),
    ('column_transformer', column_transformer)])

models = [
    LogisticRegression(solver='lbfgs', max_iter=1000),
    RandomForestClassifier(),
    SVC()]

X = df.drop('price_category', axis=1)
y = df['price_category']

best_score = 0
best_pipe = None
for model in models:
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
    score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
    if score.mean() > best_score:
        best_score = score.mean()
        best_pipe = pipe
best_pipe.fit(X, y)

print(f'Best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

with open('cars_pipe.pkl', 'wb') as file:
    dill.dump({'model': best_pipe, 'metadata': {
            'name': 'Car price category prediction model',
            'author': 'Artur',
            'version': 1,
            'date': datetime.datetime.now(),
            'type': type(best_pipe.named_steps["classifier"]).__name__,
            'accuracy': best_score}}, file)

with open("cars_pipe.pkl", "rb") as file:
    loaded_model = dill.load(file)

test_row = X.iloc[[0]]
print(loaded_model["model"].predict(test_row))