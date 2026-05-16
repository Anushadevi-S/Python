import os
import pandas as pd
from sklearn import linear_model
from word2number import w2n
import math

##df_hiringcsv = pd.read_csv("Python/ML/1_linear_reg_multivariate/hiring.csv")
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "hiring.csv")
df_hiringcsv = pd.read_csv(csv_path)
print(df_hiringcsv)
df_hiringcsv.experience = df_hiringcsv.experience.fillna('Zero')

df_hiringcsv.experience = df_hiringcsv.experience.apply(w2n.word_to_num)
score_mean= math.floor(df_hiringcsv['test_score(out of 10)'].mean())
df_hiringcsv['test_score(out of 10)'] = df_hiringcsv['test_score(out of 10)'].fillna(score_mean)
print(df_hiringcsv)

model = linear_model.LinearRegression()
model.fit(df_hiringcsv[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df_hiringcsv['salary($)'])
print(model.predict([[2, 9, 6]]))
print(model.predict([[12, 10, 10]]))
print(model.predict([[0, 8, 6]]))