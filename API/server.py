from fastapi import FastAPI, Request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

app_name = 'PredictChurn'
app_version = '1.0'
app = FastAPI(
    title=app_name,
    description='Model to predict churn in a bank',
    version=app_version
)


# Auxiliar feature engineering
def product_utilization_rate_by_year(row):
    number_of_products = row.NumOfProducts
    tenure = row.Tenure
    
    if number_of_products == 0:
        return 0
    
    if tenure == 0:
        return number_of_products
    
    rate = number_of_products / tenure
    return rate


# Auxiliar feature engineering
def credit_score_table(row):
    credit_score = row.CreditScore
    if credit_score >= 300 and credit_score < 500:
        return "Very_Poor"
    elif credit_score >= 500 and credit_score < 601:
        return "Poor"
    elif credit_score >= 601 and credit_score < 661:
        return "Fair"
    elif credit_score >= 661 and credit_score < 781:
        return "Good"
    elif credit_score >= 851:
        return "Top"
    elif credit_score >= 781 and credit_score < 851:
        return "Excellent"
    elif credit_score < 300:
        return "Deep"


# Auxiliar feature engineering
def countries_monthly_average_salaries(row):
    fr = 3696    
    de = 4740
    sp = 2257
    salary = row.EstimatedSalary / 12
    country = row.Geography              # Germany, France and Spain
    
    if country == 'Germany':
        return salary / de
    elif country == "France":
        return salary / fr
    elif country == "Spain": 
        return salary / sp


def feature_engineering(df):
    df_fe = df.copy()
    
    # balance_salary_rate
    balance_salary_rate = 'balance_salary_rate'
    df_fe[balance_salary_rate] = df_fe.Balance / df_fe.EstimatedSalary
    
    # product_utilization_rate_by_year
    df_fe = df_fe.assign(product_utilization_rate_by_year=df_fe.apply(lambda x: product_utilization_rate_by_year(x), axis=1)) 
        
    # tenure_rate_by_age - standardize and ratio by customer age, excluding adolescent period!
    tenure_rate_by_age = 'tenure_rate_by_age'
    df_fe[tenure_rate_by_age] = df_fe.Tenure / (df_fe.Age - 17)
    
    # credit_score_rate_by_age - standardize and ratio by age, excluding adolescent period!
    credit_score_rate_by_age = 'credit_score_rate_by_age'
    df_fe[credit_score_rate_by_age] = df_fe.CreditScore / (df_fe.Age - 17)
   
    # credit_score_rate_by_salary - ratio by salary
    credit_score_rate_by_salary = 'credit_score_rate_by_salary'
    df_fe[credit_score_rate_by_salary] = df_fe.CreditScore / (df_fe.EstimatedSalary)
    
    # feature engineering add - credit_score_table
    df_fe = df_fe.assign(credit_score_table=df_fe.apply(lambda x: credit_score_table(x), axis=1))
    
    # feature engineering add - countries monthly average salaries
    df_fe = df_fe.assign(countries_monthly_average_salaries = df_fe.apply(lambda x: countries_monthly_average_salaries(x), axis=1)) 
    
    return df_fe


def data_encoding(df):
    df_model = df.copy()
    
    # >>>> Categorical columns <<<<<
    non_encoding_columns = ["Geography","HasCrCard","IsActiveMember","Gender","NumOfProducts","Tenure","credit_score_table"]
    df_non_encoding = df_model[non_encoding_columns]
    df_model = df_model.drop(non_encoding_columns,axis=1)
    
    df_encoding = df_non_encoding.copy()
    
    encoder = LabelEncoder()
    df_encoding["gender_category"] = encoder.fit_transform(df_non_encoding.Gender)
    df_encoding["country_category"] = encoder.fit_transform(df_non_encoding.Geography)
    df_encoding["credit_score_category"] = encoder.fit_transform(df_non_encoding.credit_score_table)

    df_encoding.reset_index(drop=True, inplace=True)
    df_model.reset_index(drop=True, inplace=True)
    df_model = pd.concat([df_model,df_encoding],axis=1)

    df_model = df_model.drop(["Geography","Gender","CustomerId","Surname","credit_score_table","CreditScore","EstimatedSalary"],axis=1)
    df_model = df_model.reset_index()
    df_model = df_model.drop('index',axis=1)
    
    df_model.loc[df_model.HasCrCard == 0, 'credit_card_situation'] = -1
    df_model.loc[df_model.IsActiveMember == 0, 'is_active_member'] = -1
    
    return df_model


# Load model
def initialize():
    with open('best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    return model

model = initialize()

@app.get('/')
async def home_page():
    """Check app health"""
    return{app_name:app_version}

@app.post('/single_predict')
async def single_predict(request: Request):
    body = await request.json()
    df = pd.DataFrame([body])
    
    # Data modeling
    df_fe = feature_engineering(df)
    df_encoded = data_encoding(df_fe)
    df_encoded.drop(['credit_card_situation', 'is_active_member'], axis=1, inplace=True)
    print(df_encoded)
    X_test = df_encoded.to_numpy()
    print(X_test)
    print(X_test.shape)
    
    # Predict
    y_pred = model.predict(X_test)
    print(f"Churn = {y_pred}")
    
    return bool(y_pred[0])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)