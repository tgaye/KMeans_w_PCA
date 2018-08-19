import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

df_train = pd.read_csv('survey.csv')

# responses grouped by instructor (3)
def hist_total_response_by_instr():
    plt.figure(figsize=(20, 6))
    sns.countplot(x='instr', data=df_train)
    plt.show()
hist_total_response_by_instr()

# responses grouped by class
def hist_total_response_by_class():
    plt.figure(figsize=(20, 6))
    sns.countplot(x='class', data=df_train)
    plt.show()
hist_total_response_by_class()

df_train[(df_train['class']==3) | (df_train['class']==13)]['instr'].unique()
df_train[df_train['class']==13]['instr'].unique()
df_train[df_train['class']==3]['instr'].unique()

# boxplot for each question
def boxplot_rating_by_question():
    plt.figure(figsize=(20, 20))
    sns.boxplot(data=df_train.iloc[:, 5:31])
    plt.show()
boxplot_rating_by_question()

# find mean response
len(df_train[df_train['Q13'] == 1])
len(df_train[df_train['Q14'] == 1])

X_questions = df_train.iloc[:,5:33]
question_means = X_questions.mean(axis = 0) # (28 x 1)
question_means = question_means.to_frame('mean')
question_means.reset_index(level=0, inplace=True)

# respones grouped by question
def hist_question_means():
    plt.figure()
    sns.barplot(x="index", y="mean", data=question_means)
    plt.ylim(1,5)
    plt.show()
hist_question_means()

grand_mean = question_means.mean()
print('Mean response: {}'.format(grand_mean))

std_by_questions = question_means.std()
print('Mean S.D: {}'.format(std_by_questions))



