import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
sns.set(style='whitegrid')
#%matplotlib inline #(code for jupyter note book)
# load both data sets
# the files are using semicolons as separators
data_rw = pd.read_csv('./winequality-red.csv', sep=';')
data_ww = pd.read_csv('./winequality-white.csv', sep=';')
data_rw.shape
data_ww.shape
data_rw.head()
data_rw.info()
data_ww.head()
data_ww.info()
#Checking for nulls
data_ww.isnull().any()
data_rw.isnull().any()
#Prepare the Data (new columns and merge dataframes)
#Add the column "type" (red or white) to both dataframes
data_rw.insert(0, 'type', 'red')
data_ww.insert(0, 'type', 'white')
data_rw.head()
data_ww.head()
#Merge red and white wine dataframes to one dataframe.
wines = data_rw.append(data_ww, ignore_index=True)
wines.shape
wines.info()
wines['quality class'] = wines.quality.apply(lambda q: 'low' if q <= 5 \
                                             else 'high' if q > 7 else 'medium')
wines.head()
wines.info()
wines.apply(lambda c: [c.unique()])
wines.apply(lambda c: c.unique().shape[0])
wines.dtypes.value_counts()
wines.describe()
#Save the dataframe to the csv file for future use:
wines.to_csv('./winesdz.csv', index=False)
#Descriptive Statistics by Type of the Wine
wines.columns
round(wines.loc[wines.type == 'red', wines.columns].describe(),2).T
round(wines.loc[wines.type == 'white', wines.columns].describe(),2).T
# create descriptive statistics dataframes for each type of wine
rws = round(wines.loc[wines.type == 'red', wines.columns].describe(),2).T
wws = round(wines.loc[wines.type == 'white', wines.columns].describe(),2).T
# concatenate those two dataframes
pd.concat([rws, wws], axis=1, keys=['Red Wine', 'White Wine'])
#Descriptive Statistics by Quality of the Wine
# create descriptive statistics dataframes for each quality bucket
lqs = round(wines.loc[wines['quality class'] == 'low', wines.columns].describe(),2).T
mqs = round(wines.loc[wines['quality class'] == 'medium', wines.columns].describe(),2).T
hqs = round(wines.loc[wines['quality class'] == 'high', wines.columns].describe(),2).T
# concatenate those three dataframes
pd.concat([lqs, mqs, hqs], axis=1, keys=['Low Quality Wine', 'Medium Quality Wine', 'High Quality Wine'])
#Transpose it back to see all data in the notebook
pd.concat([lqs, mqs, hqs], axis=1, keys=['Low Quality Winw', 'Medium Quality Wine', 'High Quality Wine']).T
#Exploratory Data Analysis
#Distributions of wines per type and quality ratings

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
f.suptitle('Wine Type vs Quality', fontsize=14)
f.subplots_adjust(top=0.85, wspace=0.3)

sns.countplot(x='quality',
              data=wines[wines.type == 'red'],
              color='red',
              edgecolor='black',
              ax=ax1)
ax1.set_title('Red Wine')
ax1.set_xlabel('Quality')
ax1.set_ylabel('Frequency',size=12)
ax1.set_ylim([0, 2300])

sns.countplot(x='quality',
              data=wines[wines.type == 'white'],
              color='palegreen',
              edgecolor='black',
              ax=ax2)
ax2.set_title("White Wine")
ax2.set_xlabel("Quality")
ax2.set_ylabel("Frequency") 
ax2.set_ylim([0, 2300])
#Distributions of wines per type and quality classes

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
f.suptitle('Wine Type vs Quality Classes', fontsize=14)
f.subplots_adjust(top=0.85, wspace=0.3)

sns.countplot(x='quality class',
              data=wines[wines.type == 'red'],
              color='red',
              order=['low','medium','high'],
              edgecolor='black',
              ax=ax1)
ax1.set_title('Red Wine')
ax1.set_xlabel('Quality Class')
ax1.set_ylabel('Frequency',size=12)
ax1.set_ylim([0, 3200])

sns.countplot(x='quality class',
              data=wines[wines.type == 'white'],
              color='palegreen',
              order=['low','medium','high'],
              edgecolor='black',
              ax=ax2)
ax2.set_title("White Wine")
ax2.set_xlabel("Quality Class")
ax2.set_ylabel("Frequency",size=12) 
ax2.set_ylim([0, 3200])
#Preparing the dataframe and the type and quality class target variables
# to randomize data points we should re-shuffle records 
wines = wines.sample(frac=1, random_state=77).reset_index(drop=True)
# create a label (category) encoder object
le = LabelEncoder()

# fit the encoder to dataframe column and return encoded labels (transferred to integers)
y_type = le.fit_transform(wines.type.values) # 0 - Red ; 1 - White

# add a new column "color" with normalized labels
# it will be used later as the wine type target variable (1st research question)
wines['color'] = y_type
type(y_type)
numpy.ndarray
wines.head()
wines.info()
wines.color.unique()
array([0, 1], dtype=int64)
#Preparing the target variable for the 2nd research question (predicting the quality class)
# convert non-numeric quality class labels to numeric labels
# according to the dictionary's mapping
qcl = {'low':0, 'medium': 1, 'high': 2}
y_qclass = wines['quality class'].map(qcl)
y_qclass.head()
type(y_qclass)
#pandas.core.series.Series
y_qclass.unique()
#array([1, 0, 2], dtype=int64)
#Checking correlations based on the type of wines (red or white)
wcorr = wines.corr()
# sort features in order of their correllation with type of wines (column "color")
sort_corr_cols = wcorr.color.sort_values(ascending=False).keys()
sort_corr_t = wcorr.loc[sort_corr_cols,sort_corr_cols]
sort_corr_t
# heatmap plot for correlations
plt.figure(figsize=(13.5,11.5))
sns.heatmap(sort_corr_t,
            annot=True,
            annot_kws=dict(fontsize=14),
            square=True,
            fmt='.2f',
            cmap='coolwarm')
plt.title('Wine Attributes Correlations by Wine Type',
          fontsize=14,
          fontweight='bold',
          pad=10)
plt.xticks(rotation=50,fontsize=12,fontweight='bold')
plt.yticks(fontsize=12,fontweight='bold')
#Plot pairplot for wine attributes by type of wine

g = sns.pairplot(wines,
                 hue='type',
                 palette={'red' : 'red', 'white' : 'palegreen'},
                 plot_kws=dict(edgecolor='b', linewidth=0.5))

fig = g.fig
fig.subplots_adjust(top=0.95, wspace=0.2)
fig.suptitle('Wine Attributes by Wine Types',
             fontsize=26,
             fontweight='bold')

# save the plot for easier analyzing out of notebook
g.savefig('./Figures/pairplot1.png')
#Checking correlations based on the quality of wines
# sort features in order of their correllations with quality of wines (column "quality")
sort_corr_cols = wcorr.quality.sort_values(ascending=False).keys()
sort_corr_q = wcorr.loc[sort_corr_cols,sort_corr_cols]
# heatmaps plot for correlations
plt.figure(figsize=(13.5,11.5))
sns.heatmap(sort_corr_q,
            annot=True,
            annot_kws=dict(fontsize=14),
            square=True,
            fmt='.2f',
            cmap='coolwarm')
plt.title('Wine Attributes Correlations by Wine Quality Classes',
          fontsize=14,
          fontweight='bold',
          pad=10)

plt.xticks(rotation=50,fontsize=12,fontweight='bold')
plt.yticks(fontsize=12,fontweight='bold')
# drop the "quality" column and add previously defined y_qclass
# for easier plotting
wines_pq = wines.drop('quality', axis=1)
wines_pq['q_class'] = y_qclass
wines_pq.head()
#Plot pairplot for wine attributes by quality classes of wine
g = sns.pairplot(wines_pq,
                 hue='quality class',
                 palette={'high' : 'coral', 'medium' : 'palegreen', 'low' : 'dodgerblue'},
                 plot_kws=dict(edgecolor='b', linewidth=0.5))

fig = g.fig
fig.subplots_adjust(top=0.95, wspace=0.2)
fig.suptitle('Wine Attributes by Wine Quality Classes',
             fontsize=26,
             fontweight='bold')
# save the plot for easier analyzing out of notebook
g.savefig('./Figures/pairplot2.png')
#Let's see the relation of wine quality vs all wine parameters by color of wine
for f in wines.drop(['type','quality','quality class','color'],axis=1).columns:
    print(f)
for attr in wines.drop(['type','quality','quality class','color'],axis=1).columns:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    f.suptitle('Wine Type - Quality - '+ attr, fontsize=14)
    f.subplots_adjust(top=0.80, wspace=0.3)

    sns.boxplot(x='quality',
                y=attr,
                hue='type',
                data=wines,
                palette={'red' : 'coral', 'white':'palegreen'},
                ax=ax1)
    ax1.set_xlabel('Quality')
    ax1.set_ylabel(attr,size=12)
    ax1.legend(title='Wine Type',bbox_to_anchor=(1.1,1.15))
    
    sns.boxplot(x='quality class',
                y=attr,
                hue='type',
                data=wines,
                order=['low','medium','high'],
                palette={'red' : 'coral', 'white':'palegreen'},
                ax=ax2)
    ax2.set_xlabel("Quality Class")
    ax2.set_ylabel(attr) 
    ax2.legend(loc=1,title='Wine Type',bbox_to_anchor=(1.1,1.15))
     # lmplot for density vs. alcohol by wine type and quality classes
g = sns.lmplot(x='alcohol',
               y='density',
               col='type',
               col_order=['red','white'],
               hue='quality class',
               hue_order=['low','medium','high'],
               data=wines,
               palette=sns.light_palette('navy', 4),
               scatter_kws=dict(alpha=0.95,edgecolor="k", linewidth=0.5),
               fit_reg=True,
               legend=False)
fig = g.fig 
fig.subplots_adjust(top=0.85, wspace=0.3)
fig.suptitle('Wine Type - Density - Alcohol - Quality', fontsize=14)
g.add_legend(title='Wine Quality Class')    
#lmplot for total sulfur dioxide vs. residual sugar by wine type and quality classes
g = sns.lmplot(x='residual sugar',
               y='total sulfur dioxide',
               col='type',
               col_order=['red','white'],
               hue='quality class',
               hue_order=['low','medium','high'],
               data=wines,
               palette=sns.light_palette('green', 3),
               scatter_kws=dict(alpha=0.9,edgecolor="k", linewidth=0.5),
               fit_reg=False,
               legend=False)
fig = g.fig 
fig.subplots_adjust(top=0.85, wspace=0.3)
fig.suptitle('Wine Type - Sulfur Dioxide - Residual Sugar - Quality', fontsize=14)
g.add_legend(title='Wine Quality Class')   
#lmplot for total sulfur dioxide vs. residual sugar by wine type
g = sns.lmplot(x='residual sugar',
               y='total sulfur dioxide',
               hue='type',
               hue_order=['white','red'],
               data=wines,
               palette = {'red': 'coral', 'white':'palegreen'},
               scatter_kws=dict(alpha=0.8,edgecolor="k", linewidth=0.5),
               fit_reg=True,
               legend=False)
fig = g.fig 
fig.subplots_adjust(top=0.85, wspace=0.3)
fig.suptitle('Wine Type - Sulfur Dioxide - Residual Sugar', fontsize=14)
g.add_legend(title='Wine Type')   
#Predicting Wine Type (Red or White)
#Let's check our dataframe again:
wines.head()     
#Extract Features and Target
features = wines.drop(['type','quality','quality class','color'], axis=1).columns
X = wines[features].copy()
X.head()
y = wines.color.copy()
y.head()
wines.groupby('color').color.count()
#Data distribution by wine type
#wine distribution based on type
sns.countplot(x='type',
              data=wines,
              edgecolor='black',
              palette={'red':'red','white':'palegreen'})

plt.title('Wine Sample Distribution by Type',
          fontsize=14,
          pad=10)

#logistic Regression
#Split dataset into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=77,stratify=y)
#Declare a modeling pipeline with standard scaler

# Create a modeling pipeline with the standard scaler and a model
pipeline = Pipeline([
    ('scl',StandardScaler()),
    ('lr',LogisticRegression(random_state=77))
])
#Declare hyperparameters to tune

# List tunable parameters
print(pipeline.get_params())
# Declare parameters to tune
param_grid = {
    'lr__C': [0.1,1, 10, 100],
    'lr__tol': [0.001,0.0001]
}
#Sklearn cross-validation with pipeline
clf = GridSearchCV(pipeline, param_grid, cv=10)
# Fit and tune the model
clf.fit(X_train,y_train)
# list the best set of parametars found by using CV
clf.best_params_
clf.best_estimator_
#Predictions and Evaluations
# predict a new set of data
y_pred = clf.predict(X_test)
# evaluate performance of the classifier
target_names = ['red','white']
print(classification_report(y_test,y_pred,target_names=target_names),'\n')
print(confusion_matrix(y_test,y_pred))
#Plot heatmaps of confusion matrix: without and with normalization

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
f.suptitle('Logistic Regression', fontsize=14)
f.subplots_adjust(top=0.85, wspace=0.3)

# confusion matrix without normalization
mat = confusion_matrix(y_test,y_pred)
sns.heatmap(mat,
            annot=True,
            fmt='d',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax1)

ax1.set_xticklabels(labels=['red','white'])
ax1.set_yticklabels(labels=['red','white'])
ax1.set_title('Confusion Matrix w/o Normalization')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

# normalized confusion matrix 
matn = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(matn,
            annot=True,
            fmt='.2f',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax2)
 
ax2.set_xticklabels(labels=['red','white'])
ax2.set_yticklabels(labels=['red','white'])
ax2.set_title('Normalized Confusion Matrix')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')
# measure accuracy of the classifier
accuracy_score(y_true = y_test, y_pred = y_pred)
#predicting quality class of wine
wines.head()
#Extract Features and Target
features = wines.drop(['type','quality','quality class','color'], axis=1).columns
X = wines[features].copy()
X.head()
# earlier we prepared numerical target variable "y_class". Let's use it now
y = y_qclass
y.head()
y.value_counts()
#Data Distribution by quality class of wine
# wine distribution based on quality class
# low:0-5; medium:6-7; high:8-9
sns.countplot(x='quality class',
              data=wines,
              edgecolor='black',
              order=['low','medium','high'])

plt.title('Wine Sample Distribution by Quality Class',
          fontsize=14,
          pad=10)

#three classifiers: Decision Tree, Random Forest and Support Vector Machine.
#Decision_tree
#Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=77,stratify=y)
#Declare a modeling pipeline with standard scaler
# Create a modeling pipeline with the standard scaler and a model
pipeline = Pipeline([
    ('scl',StandardScaler()),
    ('dtree',DecisionTreeClassifier(random_state=77))
])
#Declare hyperparameters to tune
# List tunable parameters
print(pipeline.get_params())
# Declare parameters to tune
param_grid = {
    'dtree__min_samples_leaf': [2, 3, 4, 6],
    'dtree__max_depth': [8, 9, 10, 12, 13],
    'dtree__criterion': ['gini','entropy'],
    'dtree__class_weight': ['balanced', None]
}
#Sklearn cross-validation with pipeline
clf = GridSearchCV(pipeline, param_grid, cv=10)
# Fit and tune the model
clf.fit(X_train,y_train)
# list the best set of parametars found by using CV
clf.best_params_
clf.best_estimator_
#Predictions and Evaluations
# predict a new set of data
y_pred = clf.predict(X_test)
# evaluate performance of the classifier
target_names = ['low','medium','high']
print(classification_report(y_test,y_pred,target_names=target_names),'\n')
print(confusion_matrix(y_test,y_pred))
# measure accuracy of the classifier
accuracy_score(y_true = y_test, y_pred = y_pred)
#Plot heatmaps of confusion matrix: without and with normalization
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
f.suptitle('Decision Tree Classifier', fontsize=14)
f.subplots_adjust(top=0.85, wspace=0.3)

# confusion matrix without normalization
mat = confusion_matrix(y_test,y_pred)
sns.heatmap(mat,
            annot=True,
            fmt='d',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax1)

ax1.set_xticklabels(labels=['low','medium','high'])
ax1.set_yticklabels(labels=['low','medium','high'])
ax1.set_title('Confusion Matrix w/o Normalization')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

# normalized confusion matrix 
matn = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(matn,
            annot=True,
            fmt='.2f',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax2)
 
ax2.set_xticklabels(labels=['low','medium','high'])
ax2.set_yticklabels(labels=['low','medium','high'])
ax2.set_title('Normalized Confusion Matrix')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')
# save for later
matn_dtc = matn
#Random_forest classifier
#Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=77,stratify=y)
#Declare a modeling pipeline with standard scaler
# Create a modeling pipeline with the standard scaler and a model
pipeline = Pipeline([
    ('scl',StandardScaler()),
    ('rfc',RandomForestClassifier(random_state=77))
])
#Declare hyperparameters to tune
# List tunable parameters
print(pipeline.get_params())
# Declare parameters to tune
param_grid = {
   # 'rfc__min_samples_leaf': [1, 2, 3],
    'rfc__min_samples_split': [2, 3, 4],
    'rfc__n_estimators': [150, 175, 200 ],
    'rfc__max_depth': [20, 40, None],
    'rfc__criterion': ['gini','entropy'],
    'rfc__class_weight': ['balanced', None]
}
#Sklearn cross-validation with pipeline
clf = GridSearchCV(pipeline, param_grid, cv=10)
# Fit and tune the model
clf.fit(X_train,y_train)
# list the best set of parametars found by using CV
clf.best_params_
clf.best_estimator_
#Predictions and Evaluations
# predict a new set of data
y_pred = clf.predict(X_test)
# # evaluate performance of the classifier
target_names = ['low','medium','high']
print(classification_report(y_test,y_pred,target_names=target_names),'\n')
print(confusion_matrix(y_test,y_pred))
# measure accuracy of the classifier
accuracy_score(y_true = y_test, y_pred = y_pred)
#Plot heatmaps of confusion matrix: without and with normalization
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
f.suptitle('Random Forest Classifier', fontsize=14)
f.subplots_adjust(top=0.85, wspace=0.3)
# confusion matrix without normalization
mat = confusion_matrix(y_test,y_pred)
sns.heatmap(mat,
            annot=True,
            fmt='d',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax1)

ax1.set_xticklabels(labels=['low','medium','high'])
ax1.set_yticklabels(labels=['low','medium','high'])
ax1.set_title('Confusion Matrix w/o Normalization')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')
matn = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(matn,
            annot=True,
            fmt='.2f',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax2)
 
ax2.set_xticklabels(labels=['low','medium','high'])
ax2.set_yticklabels(labels=['low','medium','high'])
ax2.set_title('Normalized Confusion Matrix')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')
# save for later
matn_rfc = matn
#support_vector machine
#Split dataset into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=77,stratify=y)
#Declare a modeling pipeline with standard scaler
# Create a modeling pipeline with the standard scaler and a model
pipeline = Pipeline([
    ('scl',StandardScaler()),
    ('svc',SVC(random_state=77))
])
#Declare hyperparameters to tune
# List tunable parameters
print(pipeline.get_params())
param_grid = {
    'svc__C': [0.08, 0.1, 1, 10],
    'svc__gamma': [5, 1, 0.1, 0.01]
}
#Sklearn cross-validation with pipeline
clf = GridSearchCV(pipeline, param_grid, cv=10)
# Fit and tune the model
clf.fit(X_train,y_train)
# list the best set of parametars found by using CV
clf.best_params_
clf.best_estimator_
#Predictions and Evaluations
# predict a new set of data
y_pred = clf.predict(X_test)
# evaluate performance of the classifier
target_names = ['low','medium','high']
print(classification_report(y_test,y_pred,target_names=target_names),'\n')
print(confusion_matrix(y_test,y_pred))
# measure accuracy of the classifier
accuracy_score(y_true = y_test, y_pred = y_pred)
#Plot heatmaps of confusion matrix: without and with normalization

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
f.suptitle('Support Vector Machine Classifier', fontsize=14)
f.subplots_adjust(top=0.85, wspace=0.3)
# confusion matrix without normalization
mat = confusion_matrix(y_test,y_pred)
sns.heatmap(mat,
            annot=True,
            fmt='d',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax1)

ax1.set_xticklabels(labels=['low','medium','high'])
ax1.set_yticklabels(labels=['low','medium','high'])
ax1.set_title('Confusion Matrix w/o Normalization')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')
# normalized confusion matrix 
matn = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(matn,
            annot=True,
            fmt='.2f',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax2)
 
ax2.set_xticklabels(labels=['low','medium','high'])
ax2.set_yticklabels(labels=['low','medium','high'])
ax2.set_title('Normalized Confusion Matrix')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')
# save for later
matn_svc = matn
#compare normalized confusion matrices for all 3 classifiers:

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
f.suptitle('Normalized Confusion Matrices for Prediction of Wine Quality Classes', fontsize=14)
f.subplots_adjust(top=0.85, wspace=0.3)

# normalized confusion matrix for decision tree classifier
sns.heatmap(matn_dtc,
            annot=True,
            fmt='.2f',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax1)

ax1.set_xticklabels(labels=['low','medium','high'])
ax1.set_yticklabels(labels=['low','medium','high'])
ax1.set_title('Decision Tree Classifier')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

# normalized confusion matrix for random forest classifier
sns.heatmap(matn_rfc,
            annot=True,
            fmt='.2f',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax2)
 
ax2.set_xticklabels(labels=['low','medium','high'])
ax2.set_yticklabels(labels=['low','medium','high'])
ax2.set_title('Random Forest Classifier')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

# normalized confusion matrix for support vector machine classifier
sns.heatmap(matn_svc,
            annot=True,
            fmt='.2f',
            cbar=True,
            square=True,
            cmap='Oranges',
            ax=ax3)
 
ax3.set_xticklabels(labels=['low','medium','high'])
ax3.set_yticklabels(labels=['low','medium','high'])
ax3.set_title('Support Vector Machine Classifier')
ax3.set_xlabel('Predicted Label')
ax3.set_ylabel('True Label')