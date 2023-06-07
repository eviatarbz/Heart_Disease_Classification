import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



df = pd.read_csv(r'C:\Users\user\Downloads\heart_2020_cleaned (1).csv')
df['ID'].apply(lambda x: int(x))
df.head(30).drop('HeartDisease',axis=1).to_excel('check_data.xlsx',index = False)
odf =  df
def convert_age(df): #המרה של גיל מקטגורי לנומרי
    encode_AgeCategory = {'55-59':57, '80 or older':80, '65-69':67,
                          '75-79':77,'40-44':42,'70-74':72,'60-64':62,
                          '50-54':52,'45-49':47,'18-24':21,'35-39':37,
                          '30-34':32,'25-29':27}
    df['AgeCategory'] = df['AgeCategory'].apply(lambda x: encode_AgeCategory[x])
    df['AgeCategory'] = df['AgeCategory'].astype('float')
    return df
convert_age(df)


def visualization(df):#היחס בין חולים לבריאים במאגר
    fig, ax = plt.subplots(figsize=(15, 8))
    df.value_counts().plot(kind='pie', autopct='%.1f%%', labels=df.unique(), explode=(0, 0.05)
    , colors=['#27ae60', '#c0392b'])
    ax.set_title('Having HD Ratio')
    plt.show()
    
visualization(df['HeartDisease'])

def pie_graphs(df):#גרפים המראים על חלוקה בין חולים לבריאים בכל נתון בינארי 
    binary_cols = ['HeartDisease','Sex','Smoking','AlcoholDrinking','Stroke','Asthma', 'DiffWalking','PhysicalActivity','KidneyDisease','SkinCancer']
    
    
    for i in range(1, len(binary_cols)):
        fig = plt.figure(figsize=(8,4), dpi=80)
        
        
        ax1 = plt.subplot(1,2,1)
        grouped = df[df[binary_cols[i]] == (df[binary_cols[i]].unique()[0])].groupby(df['HeartDisease'])
        counts = grouped[binary_cols[i]].count()
        counts.plot(kind='pie', autopct='%.1f%%', labeldistance=None, wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white', 'width':0.35 })
        plt.gca().axes.get_yaxis().set_visible(False)   
        plt.title(f"when {binary_cols[i]} value is: {df[binary_cols[i]].unique()[0]}")   
        ax2 = plt.subplot(1,2,2)
        grouped1 = df[df[binary_cols[i]] == (df[binary_cols[i]].unique()[1])].groupby(df['HeartDisease'])
        counts1 = grouped1[binary_cols[i]].count()
        counts1.plot(kind='pie', autopct='%.1f%%', labeldistance=None, wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white', 'width':0.35 })
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.title(f"when {binary_cols[i]} value is: {df[binary_cols[i]].unique()[1]}")
        plt.suptitle("Patient with/without Heart Disease distribution by " + binary_cols[i] + " status", fontweight='bold')
        
        handles, labels = ax1.get_legend_handles_labels()
    
            
        leg = fig.legend(handles, labels, loc = 'upper right', fancybox=True)
        
            
        plt.subplots_adjust(right=0.9)
        plt.show()
        

#convert strings to numeric values


def standardize_data(new_df,df):#נירמול של המאגר
    def standardize(num,avg,std):
        s = (num - avg)/std
        return s
    for col in ['SleepTime','BMI','PhysicalHealth','MentalHealth','AgeCategory']:
        new_df[col] = np.vectorize(standardize)(new_df[col],df[col].mean(),df[col].std())
    return new_df
df = standardize_data(df,df)
ndf = df[(np.abs(df['SleepTime'])<3) & (np.abs(df['BMI'])<3)] # הוצאת חריגים
pie_graphs(ndf)
rdf = ndf.copy()


df_nominal = df.select_dtypes(include='object')
ids = ndf['ID']
rdf = rdf[rdf['ID'].isin(ids)]
nominal_cols = df_nominal.columns

numeric_cols = ndf.columns
numeric_cols = numeric_cols.drop(nominal_cols)
nominal_cols = nominal_cols.drop('HeartDisease')
numeric_cols = numeric_cols.drop(['ID'])
print((rdf['HeartDisease'].value_counts()))
X = rdf.drop(['HeartDisease','ID'],axis = 1)
Y = rdf['HeartDisease']
check_x = ndf.drop(['ID','HeartDisease'],axis = 1)
check_y = ndf['HeartDisease']





from smote_class import SMOTENC
smote_nc = SMOTENC(nominal_cols, numeric_cols, k=5)
minority_features = df_nominal.columns.get_indexer(df_nominal.columns)

X_synthetic_df = smote_nc.smotenc(check_x.head(15000),check_y.head(15000))
X_synthetic_df = pd.DataFrame(X_synthetic_df,columns = X.columns)
synthetic_df = X_synthetic_df
synthetic_df['HeartDisease'] = 'Yes'

combined_df = (pd.concat((ndf.head(15000),synthetic_df),axis=0)).drop('ID',axis=1)
combined_df = combined_df.reset_index(drop=True)
pie_graphs(combined_df)

X_smoted = combined_df.drop(['HeartDisease'], axis =1)
y_smoted = combined_df['HeartDisease']

y_smoted = y_smoted.apply(lambda x: 0 if x=='No' else 1)
one_hot = pd.get_dummies(X_smoted[nominal_cols],dtype = float)
X_smoted = pd.concat([X_smoted.select_dtypes(include=["float64", "int64"]), one_hot], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_smoted, y_smoted, test_size=0.25, random_state=42)


from logistic_regression_class import LogisticRegression
model_lgr= LogisticRegression()
model_lgr.fit(X_train, y_train)

from svm_class import SVM
model_svm = SVM()
model_svm.fit(X_train.to_numpy(), y_train.to_numpy())


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators=100)
model_rf.fit(X_train.values, y_train.values)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
base_estimator = DecisionTreeClassifier(max_depth=10)
model_adaboost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, random_state=42)
model_adaboost.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier(n_neighbors=5)  # Specify the number of neighbors to consider
model_knn.fit(X_train, y_train)



from sklearn.metrics import confusion_matrix, cohen_kappa_score, recall_score, precision_score, f1_score, roc_auc_score

def print_evaluation_scores(model, X_test, y_test):
    #פונקציה שמדפיסה מדדי דיוק עבור מודל
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test,y_pred)
    
    
    print("Confusion Matrix:")
    print(cm)
    print("Kappa:", kappa)
    print("Recall:", recall)
    print("Precision:", precision)
    print("F1 Score:", f1)
    print("AUC ROC:", auc_roc)
    print("accuracy: ", accuracy)
    
    return f1    
    
    




models = [ model_rf, model_lgr, model_adaboost, model_knn, model_svm]
best_f1 = 0 
best_model = None
for i in models:
    
    print(f"Evaluation Scores for {i}:")
    f1 = print_evaluation_scores(i, X_test, y_test)
    if(f1>best_f1):
        best_f1 = f1
        best_model = i

    
print(f"best model is: {best_model}")
visualization(y_smoted)

import tkinter as tk
from tkinter import filedialog, messagebox


from GUI_class import GUI
gui = GUI(best_model,X_train.columns.to_list(),convert_age,standardize_data,odf,nominal_cols)
gui.run()



