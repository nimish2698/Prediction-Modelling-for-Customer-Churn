#%%
# Importing the warnings package
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%%
# Importing the pandas package
import pandas as pd

# Loading dataset in python
file_path = 'C:\\Nimish\\Imp docs\\University of Surrey\\Semester 2\\AI-ML & DV\\Project\\bob-wakefield-call-center-data\\data\\customer_data_edited.csv'
# Reading the CSV file into a DataFrame
dataframe = pd.read_csv(file_path)
new_dataframe = dataframe.drop(['recordid', 'customer_id'], axis=1)

# Displaying the attribute names of the dataframe
print('\n\n\n-----------------------------------------------------------------------')
print('Attribute Names of the dataframe')
print(new_dataframe.columns)
print('----------------------------------------------------------------------- \n\n\n')

# Displaying the top 5 observations of the dataframe to verify that the data has been read correctly
print('----------------------------------------------------------------------------')
print('Top 5 observations of the dataframe')
print(new_dataframe.head())
print('----------------------------------------------------------------------------\n\n\n')

# Displaying the last 5 observations of the dataframe to verify that the data has been read correctly
print('----------------------------------------------------------------------------')
print('Last 5 observations of the dataframe')
print(new_dataframe.tail())
print('----------------------------------------------------------------------------\n\n\n')

# Displaying the types and information about data
print('----------------------------------------------------------------------------')
print('Types and Information about the dataframe')
print(new_dataframe.info())
print('----------------------------------------------------------------------------\n\n\n')

#%%
# ----------------------------------------
# Missing value identification
# ----------------------------------------
# Missing Values calculation
ms = new_dataframe.isnull().sum()
# Calculating the percentage of missing values in each column
ms_percentage = (new_dataframe.isnull().sum()/(len(new_dataframe)))*100
# Combining the missing value information into one dataframe
missing_data_info = pd.DataFrame({'Total Missings': ms, 'Percentage': ms_percentage})
# Printing the missing value information on screen
print('----------------------------------------')
print('Missing Data Information')
print(missing_data_info)
print('----------------------------------------\n\n\n')

#%%
# ----------------------------
# Data Encoding - One Hot Encoding
# ----------------------------
# Displaying the shape of the Data
dataframe_shape = new_dataframe.shape
# Number of rows
dataframe_rows = new_dataframe.shape[0]
# Number of columns
dataframe_columns = new_dataframe.shape[1]
# Separate the features only for data encoding
dataframe_features_only = new_dataframe.drop(['state',], axis = 1)
# Performing one-hot encoding, OHE = one-hot encoding
OHE_dataframe = pd.get_dummies(dataframe_features_only)
# Converting True/False to 1/0
numeric_dataframe = dataframe_features_only.astype(int)
# Concatenating the dropped feature of numeric_dataframe after encoding
# prepared_dataframe = pd.concat([numeric_dataframe, new_dataframe['state']], axis=1)

#%%
# -------------------
# Data Summary Statistics
# -------------------
# Statistical summary of data belonging to numerical datatype such as int, float
data_stat = numeric_dataframe.describe().T
print('-------------------------------------------------------------------------------')
print('Data Summary Statistics')
print('-------------------------------------------------------------------------------')
print(data_stat)
print('-------------------------------------------------------------------------------\n\n\n')
# Displaying a statistical summary of the data
data_stat_all = numeric_dataframe.describe(include='all').T

#%%
# ========================================
# EDA Univariate Analysis
# ========================================

# ----------------------------------------
# Plotting the histogram for the attributes
# ----------------------------------------

# Importing the package
import matplotlib.pyplot as plt

# Getting the list of column names
columns = numeric_dataframe.columns[0:19]
# Setting the number of rows and columns for subplots
num_rows = 5
num_cols = 5
# Creating a new figure and setting its size
plt.figure(figsize=(15, 15))

# Iterating each column and plotting a histogram for the same
for i, column in enumerate(columns, 1):
    plt.subplot(num_rows, num_cols, i)
    numeric_dataframe[column].hist()
    plt.title(column)

# Adjust layout
plt.tight_layout()
# Show the plot
plt.show()

# -----------------------------
# Boxplot
# -----------------------------

# Getting the list of column names
columns = numeric_dataframe.columns
# Setting the number of rows and columns for subplots
num_rows = 5
num_cols = 5
# Creating a new figure and setting its size
plt.figure(figsize=(15, 15))

for i, Attribute in enumerate(numeric_dataframe.columns[0:19],1):
    plt.subplot(num_rows, num_cols, i)
    # Creating plot
    plt.boxplot(numeric_dataframe[Attribute])
    plt.title(Attribute)

# show plot
plt.show()

# Calculating the counts of each churn category
churn_counts = dataframe['churn'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(churn_counts, labels=['Not Churned', 'Churned'], autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e'])
plt.title('Customer Churn Distribution')
plt.show()

#%%
# ========================================
# EDA Bivariate Analysis
# ========================================

# -----------------------------
# Scatter plot
# -----------------------------

# Creating scatter plots

plt.scatter(numeric_dataframe['total_day_minutes'], numeric_dataframe['total_day_charge'])
plt.xlabel('Total Day Minutes')
plt.ylabel('Total Day Charge')
plt.title('Bivariate Scatter Plot - Relationship between Total Day Minutes and Day Charge')
plt.show()

plt.scatter(numeric_dataframe['total_eve_minutes'], numeric_dataframe['total_eve_charge'])
plt.xlabel('Total Evening Minutes')
plt.ylabel('Total Evening Charge')
plt.title('Bivariate Scatter Plot - Relationship between Total Evening Minutes and Evening Charge')
plt.show()

# plt.scatter(numeric_dataframe['number_customer_service_calls'], numeric_dataframe['churn'])
# plt.xlabel('number_customer_service_calls')
# plt.ylabel('Churn')
# plt.title('Bivariate Scatter Plot - Number of Customer Service Calls vs. Churn')
# plt.show()

plt.scatter(numeric_dataframe['total_intl_minutes'], numeric_dataframe['total_intl_charge'])
plt.xlabel('Total International Minutes')
plt.ylabel('Total International Charge')
plt.title('Bivariate Scatter Plot - Relationship between Total International Minutes and International Charge')
plt.show()

#%%

# ========================================
# EDA Multivariate Analysis
# ========================================

# -----------------------------
# Correlation matrix plot
# -----------------------------

# Importing the package

import seaborn as sns

plt.figure(figsize=(20,15))
sns.heatmap(numeric_dataframe.drop(['churn'],axis=1).corr(), annot = True, vmin = -1, vmax = 1)
plt.show()
sns.heatmap(numeric_dataframe.corr(), annot = True, vmin = -1, vmax = 1)
plt.show()
#%%
# ========================================
# Class Distribution
# ========================================

print("-----------------------------------------------")
print("Class Distribution")
print("-----------------------------------------------")
class_distribution = numeric_dataframe['churn'].value_counts()
print("Class - no -> " + "{:.3f}".format((class_distribution[0]/dataframe_rows)*100)+ " %")
print("Class - yes -> " + "{:.3f}".format((class_distribution[1]/dataframe_rows)*100) + " % \n\n\n")

#%%

# ============================================
# KNN Classification
# ============================================

# Import the package
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,classification_report

# Separate features and target variable from the DataFrame
Features_Only = numeric_dataframe.drop(['churn'],axis=1) 
Labels_Only = numeric_dataframe['churn']
#-------------------------------
# Feature normalisation
#-------------------------------
# Initialize the Standard Scaler
scaler = StandardScaler() # MinMaxScaler()
# Fit and transform the DataFrame
Features_Only = pd.DataFrame(scaler.fit_transform(Features_Only),columns=Features_Only.columns)

print("----------------------------------------------------------------------------------------------------")
print("All Attributes Selected & Classified with KNN")

# Define the number of folds for cross-validation
num_folds = 5
# Initialize a KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Initialize a list to store the accuracy scores for each fold
Accuracy_Scores = []
# Initialize a list to store the error scores for each fold
Error_Scores = []
# Initialize a list to store the sensitivity scores for each fold
Sensitivity_Scores = []
# Initialize a list to store the specificity scores for each fold
Specificity_Scores = []
# Initialize a list to store the F1 scores for each fold
F1_Scores = []

print("----------------------------------------------------------------------------------------------------")
print("\t\tFold \t Accuracy \t\t Error \t\t Sensitivity \t Specificity \t\t F1 Score")
print("----------------------------------------------------------------------------------------------------")

# Iterate through each fold
for f, (train_index, test_index) in enumerate(kf.split(Features_Only)):
    Training_Features, Testing_Features = Features_Only.loc[train_index],Features_Only.loc[test_index]
    Training_Labels, Testing_Labels = Labels_Only.loc[train_index], Labels_Only.loc[test_index]
    # Class distribution and sampling - START
    # =============================================================================
    
    print("-----------------------------------------------")
    print("Class Distribution")
    print("-----------------------------------------------")
    class_distribution = Training_Labels.value_counts()
    print("Class - no -> " +"{:.3f}".format((class_distribution[0]/Training_Labels.size)*100) + " %")
    print("Class - yes -> " +"{:.3f}".format((class_distribution[1]/Training_Labels.size)*100) + " %")
    
    # keep a copy of Training_Features, Training_Labels for future comparsion
    Training_Features_Original = Training_Features
    Training_Labels_Original = Training_Labels
    # Apply SMOTE to balance the dataset
    # Comment/Uncomment next 2 lines for activating Sampling
    smote = SMOTE()
    Training_Features, Training_Labels = smote.fit_resample(Training_Features,Training_Labels)
    
    print("-----------------------------------------------")
    print("Class Distribution - After SMOTE Oversampling")
    print("-----------------------------------------------")
    class_distribution2 = Training_Labels.value_counts()
    print("Class - no -> " +"{:.3f}".format((class_distribution2[0]/Training_Labels.size)*100) + " %")
    print("Class - yes -> " +"{:.3f}".format((class_distribution2[1]/Training_Labels.size)*100) + " %" +"\n\n\n")
    
    # =============================================================================
    # Class distribution and sampling - FINISH
    # =============================================================================
    # Initialize 
    KNN_Classifier = KNeighborsClassifier(n_neighbors=5)
    
    # train the knn classifier
    KNN_Classifier.fit(Training_Features, Training_Labels)
    
    # Make predictions
    Predicted_Labels = KNN_Classifier.predict(Testing_Features)
    
    # Calculate confusion matrix
    Confusion_Matrix = confusion_matrix(Testing_Labels, Predicted_Labels)
    
    # Calculate accuracy for this fold
    Accuracy = accuracy_score(Testing_Labels, Predicted_Labels)
    # Append the accuracy score to the list
    Accuracy_Scores.append(Accuracy)
    
    # Calculate error for this fold
    Error = 1 - Accuracy
    # Append the error score to the list
    Error_Scores.append(Error)
    
    # Calculate sensitivity and specificity
    True_Negatives = Confusion_Matrix[0, 0]
    False_Positives = Confusion_Matrix[0, 1]
    False_Negatives = Confusion_Matrix[1, 0]
    True_Positives = Confusion_Matrix[1, 1]
    
    Sensitivity = True_Positives / (True_Positives + False_Negatives)
    # Append the sensitivity score to the list
    Sensitivity_Scores.append(Sensitivity)
    
    Specificity = True_Negatives / (True_Negatives + False_Positives)
    # Append the specificity score to the list
    Specificity_Scores.append(Specificity)
    
    # Calculate Precision and Recall from the confusion matrix
    Precision = True_Positives / (True_Positives + False_Positives)
    Recall = True_Positives / (True_Positives + False_Negatives)
    
    # Calculate F1 score
    f1 = 2 * (Precision * Recall) / (Precision + Recall)
    # Append the f1 score to the list
    F1_Scores.append(f1)
    
    print(" \t\t" + str(f) + " \t\t {:.3f}".format(Accuracy*100) + " \t\t {:.3f}".format(Error*100) + " \t\t {:.3f}".format(Sensitivity*100) + " \t\t {:.3f}".format(Specificity*100) + " \t\t {:.3f}".format(f1*100))
    
    # Classification Report can be printed/checked for each fold
    print("Classification Report:")
    print(classification_report(Testing_Labels, Predicted_Labels))

    # Printing separation line to indicate 5-fold execution completion
    print("----------------------------------------------------------------------------------------------------")
# Calculate the average accuracy across all folds
Average_Accuracy = sum(Accuracy_Scores) / len(Accuracy_Scores)
# Calculate the average error across all folds
Average_Error = sum(Error_Scores) / len(Error_Scores)
# Calculate the average sensitivity across all folds
Average_Sensitivity = sum(Sensitivity_Scores) / len(Sensitivity_Scores)
# Calculate the average specificity across all folds
Average_Specificity = sum(Specificity_Scores) / len(Specificity_Scores)
# Calculate the average f1 across all folds
Average_F1 = sum(F1_Scores) / len(F1_Scores)
# Printing the average performance scores
print("----------------------------------------------------------------------------------------------------")
print("\t\t \t Accuracy \t\t Error \t\t Sensitivity \t Specificity \t\t F1 Score")
print("----------------------------------------------------------------------------------------------------")
print("Average ->" + " \t\t {:.3f}".format(Average_Accuracy*100) + " \t\t {:.3f}".format(Average_Error*100) + " \t\t {:.3f}".format(Average_Sensitivity*100) + " \t\t{:.3f}".format(Average_Specificity*100) + " \t\t {:.3f}".format(Average_F1*100))
print("----------------------------------------------------------------------------------------------------")
# print(" Average ->" + " " + str(f) + " \t\t {:.3f}".format(Average_Accuracy*100) + " \t\t {:.3f}".
# format(Average_Error*100) + " \t\t {:.3f}".format(Average_Sensitivity*100) + " \t\t {:.3f}".format
# (Average_Specificity*100) + " \t\t {:.3f}".format(Average_F1*100))
# ------------------------------------
# Plot the average performances
# ------------------------------------

import numpy as np

# Create a new figure and set its size
plt.figure(figsize=(15, 15))

plt.subplot(2, 2, 1)
# Create a bar plot
plt.bar(['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'], np.array(Accuracy_Scores)*100)
# Add title
plt.title('Accuracy over each fold')
# Add labels
# plt.xlabel('Folds')
plt.ylabel('Accuracy (%)')
# Set y-axis limit
plt.ylim(0, 100)

plt.subplot(2, 2, 2)
# Create a bar plot
plt.bar(['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'], np.array(Sensitivity_Scores)*100)
# Add title
plt.title('Sensitivity over each fold')
# Add labels
# plt.xlabel('Folds')
plt.ylabel('Sensitivity (%)')
# Set y-axis limit
plt.ylim(0, 100)

plt.subplot(2, 2, 3)
# Create a bar plot
plt.bar(['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'], np.array(Specificity_Scores)*100)
# Add title
plt.title('Specificity over each fold')
# Add labels
# plt.xlabel('Folds')
plt.ylabel('Specificity (%)')
# Set y-axis limit
plt.ylim(0, 100)

plt.subplot(2, 2, 4)
# Create a bar plot
plt.bar(['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'], np.array(F1_Scores)*100)
# Add title
plt.title('F1 score over each fold')
# Add labels
# plt.xlabel('Folds')
plt.ylabel('F1 Score (%)')
# Set y-axis limit
plt.ylim(0, 100)

# Show the plot
plt.show()

# #%%

# # ===================================================
# # KNN Classification Performance Improvement
# # ===================================================

# # ------------------------------------------------------------------------------------
# # Feature Selection based on Correlation Matrix or Intuition you can do feature importance
# # ------------------------------------------------------------------------------------
# # Assuming df is your DataFrame and 'target' is your target column
# correlation = numeric_dataframe.corr()
# cor_target = abs(correlation['churn'])  # Absolute value of correlations
# relevant_features = cor_target[cor_target > 0.2]  # Select features with correlation above a threshold
# print(relevant_features)
# # Separate features and target variable from the DataFrame
# Features_Only_Round_2 = numeric_dataframe.drop(['account_length','area_code','voice_mail_plan','number_vmail_messages','total_day_calls',
#                                                 'total_eve_minutes','total_eve_calls','total_eve_charge',
#                                                 'total_night_minutes','total_night_minutes','total_night_calls','total_night_charge'
#                                                 ,'total_intl_minutes','total_intl_calls','total_intl_charge','churn'],axis=1)
# Labels_Only_Round_2 = numeric_dataframe['churn']
# # Initialize the Standard Scaler
# Features_Only_Round_2 = pd.DataFrame(scaler.fit_transform(Features_Only_Round_2),columns=Features_Only_Round_2.columns)
# print("\n\n\n----------------------------------------------------------------------------------------------------")
# print("Features Selected & Classified with KNN")

# # ------------------------------------------
# # Redo KNN Classification
# # ------------------------------------------

# # Define the number of folds for cross-validation
# num_folds = 5
# # Initialize a KFold object
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# # Initialize a list to store the accuracy scores for each fold
# Accuracy_Scores = []
# # Initialize a list to store the error scores for each fold
# Error_Scores = []
# # Initialize a list to store the sensitivity scores for each fold
# Sensitivity_Scores = []
# # Initialize a list to store the specificity scores for each fold
# Specificity_Scores = []
# # Initialize a list to store the F1 scores for each fold
# F1_Scores = []

# print("----------------------------------------------------------------------------------------------------")
# print("\t\tFold \t Accuracy \t\t Error \t\t Sensitivity \t Specificity \t\t F1 Score")
# print("---------------------------------------------------------------------------------------------------")

# # Iterate through each fold
# for f, (train_index, test_index) in enumerate(kf.split(Features_Only_Round_2)):
#     Training_Features, Testing_Features = Features_Only_Round_2.loc[train_index],Features_Only_Round_2.loc[test_index]
#     Training_Labels, Testing_Labels = Labels_Only_Round_2.loc[train_index],Labels_Only_Round_2.loc[test_index]
    
#     # Class distribution and sampling - START
#     # =============================================================================
    
#     print("-----------------------------------------------")
#     print("Class Distribution")
#     print("-----------------------------------------------")
#     class_distribution = Training_Labels.value_counts()
#     print("Class - no -> " +"{:.3f}".format((class_distribution[0]/Training_Labels.size)*100) + " %")
#     print("Class - yes -> " +"{:.3f}".format((class_distribution[1]/Training_Labels.size)*100) + " %")
    
#     # keep a copy of Training_Features, Training_Labels for future comparsion
#     Training_Features_Original = Training_Features
#     Training_Labels_Original = Training_Labels
#     # Apply SMOTE to balance the dataset
#     # Comment/Uncomment next 2 lines for activating Sampling
#     smote = SMOTE()
#     Training_Features, Training_Labels = smote.fit_resample(Training_Features,Training_Labels)
    
#     print("-----------------------------------------------")
#     print("Class Distribution - After SMOTE Oversampling")
#     print("-----------------------------------------------")
#     class_distribution2 = Training_Labels.value_counts()
#     print("Class - no -> " +"{:.3f}".format((class_distribution2[0]/Training_Labels.size)*100) + " %")
#     print("Class - yes -> " +"{:.3f}".format((class_distribution2[1]/Training_Labels.size)*100) + " %" +"\n\n\n")
    
#     # =============================================================================
#     # Class distribution and sampling - FINISH
#     # =============================================================================
    
#     # Initialize knn classifier
#     KNN_Classifier = KNeighborsClassifier(n_neighbors=5)
    
#     # train the knn classifier
#     KNN_Classifier.fit(Training_Features, Training_Labels)
    
#     # Make predictions
#     Predicted_Labels = KNN_Classifier.predict(Testing_Features)
    
#     # Calculate confusion matrix
#     Confusion_Matrix = confusion_matrix(Testing_Labels, Predicted_Labels)
    
#     # Calculate accuracy for this fold
#     Accuracy = accuracy_score(Testing_Labels, Predicted_Labels)
#     # Append the accuracy score to the list
#     Accuracy_Scores.append(Accuracy)
    
#     # Calculate error for this fold
#     Error = 1 - Accuracy
#     # Append the error score to the list
#     Error_Scores.append(Error)
    
#     # Calculate sensitivity and specificity
#     True_Negatives = Confusion_Matrix[0, 0]
#     False_Positives = Confusion_Matrix[0, 1]
#     False_Negatives = Confusion_Matrix[1, 0]
#     True_Positives = Confusion_Matrix[1, 1]
    
#     Sensitivity = True_Positives / (True_Positives + False_Negatives)
#     # Append the sensitivity score to the list
#     Sensitivity_Scores.append(Sensitivity)
    
#     Specificity = True_Negatives / (True_Negatives + False_Positives)
#     # Append the specificity score to the list
#     Specificity_Scores.append(Specificity)
    
#     # Calculate Precision and Recall from the confusion matrix
#     Precision = True_Positives / (True_Positives + False_Positives)
#     Recall = True_Positives / (True_Positives + False_Negatives)
    
#     # Calculate F1 score
#     f1 = 2 * (Precision * Recall) / (Precision + Recall)
#     # Append the f1 score to the list
#     F1_Scores.append(f1)
    
#     print(" \t\t" + str(f) + " \t\t {:.3f}".format(Accuracy*100) + " \t\t {:.3f}".
#     format(Error*100) + " \t\t {:.3f}".format(Sensitivity*100) + " \t\t {:.3f}".format
#     (Specificity*100) + " \t\t {:.3f}".format(f1*100))
    
#     # Classification Report can be printed/checked for each fold
#     print("Classification Report:")
#     print(classification_report(Testing_Labels, Predicted_Labels))

#     # Printing separation line to indicate 5-fold execution completion
#     print("----------------------------------------------------------------------------------------------------")
# # Calculate the average accuracy across all folds
# Average_Accuracy = sum(Accuracy_Scores) / len(Accuracy_Scores)
# # Calculate the average error across all folds
# Average_Error = sum(Error_Scores) / len(Error_Scores)
# # Calculate the average sensitivity across all folds
# Average_Sensitivity = sum(Sensitivity_Scores) / len(Sensitivity_Scores)
# # Calculate the average specificity across all folds
# Average_Specificity = sum(Specificity_Scores) / len(Specificity_Scores)
# # Calculate the average f1 across all folds
# Average_F1 = sum(F1_Scores) / len(F1_Scores)
# # Printing the average performance scores
# print("Average ->" + " \t\t {:.3f}".format(Average_Accuracy*100) + " \t\t {:.3f}".format(Average_Error*100) + " \t\t {:.3f}".format(Average_Sensitivity*100) + " \t\t{:.3f}".format(Average_Specificity*100) + " \t\t {:.3f}".format(Average_F1*100))
# print("----------------------------------------------------------------------------------------------------")

# # ------------------------------------
# # Plot the average performances
# # ------------------------------------

# # Create a new figure and set its size
# plt.figure(figsize=(15, 15))

# plt.subplot(2, 2, 1)
# # Create a bar plot
# plt.bar(['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'], np.array(Accuracy_Scores)*100)
# # Add title
# plt.title('Accuracy over each fold')
# # Add labels
# # plt.xlabel('Folds')
# plt.ylabel('Accuracy (%)')
# # Set y-axis limit
# plt.ylim(0, 100)

# plt.subplot(2, 2, 2)
# # Create a bar plot
# plt.bar(['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'], np.array(Sensitivity_Scores)*100)
# # Add title
# plt.title('Sensitivity over each fold')
# # Add labels
# # plt.xlabel('Folds')
# plt.ylabel('Sensitivity (%)')
# # Set y-axis limit
# plt.ylim(0, 100)

# plt.subplot(2, 2, 3)
# # Create a bar plot
# plt.bar(['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'], np.array(Specificity_Scores)*100)
# # Add title
# plt.title('Specificity over each fold')
# # Add labels
# # plt.xlabel('Folds')
# plt.ylabel('Specificity (%)')
# # Set y-axis limit
# plt.ylim(0, 100)

# plt.subplot(2, 2, 4)
# # Create a bar plot
# plt.bar(['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'], np.array(F1_Scores)*100)
# # Add title
# plt.title('F1 score over each fold')
# # Add labels
# # plt.xlabel('Folds')
# plt.ylabel('F1 Score (%)')
# # Set y-axis limit
# plt.ylim(0, 100)

# # Show the plot
# plt.show()


# #----------------------------------------------------------------------------------------
# # Feature Extraction based on PCA
# #----------------------------------------------------------------------------------------
# # Import the package
# from sklearn.decomposition import PCA

# # First, separate the features (X) from the target variable (if any)
# Data_without_target = numeric_dataframe.drop('churn', axis=1) # Drop the target column if present
# # Initialize PCA with the desired number of components
# pca = PCA(n_components=5) # Specify the number of components you want to retain
# # Fit PCA to the data and transform the features
# Extracted_PC_features = pca.fit_transform(Data_without_target)
# # Create a new DataFrame with the transformed features
# New_PC_features_in_dataframe = pd.DataFrame(data=Extracted_PC_features, columns=[f'PC{i+1}' for i in range(5)]) # Column names are PC1, PC2, ..., PC10
# # Combine the principal components DataFrame with the target column (if required)
# New_PC_features_with_target = pd.concat([New_PC_features_in_dataframe,numeric_dataframe['churn']], axis=1)
# # Separate features and target variable from the DataFrame
# Features_Only_Round_3 = New_PC_features_with_target.drop(['churn'],axis=1)
# Labels_Only_Round_3 = New_PC_features_with_target['churn']
# Features_Only_Round_3 = pd.DataFrame(scaler.fit_transform(Features_Only_Round_3),columns=Features_Only_Round_3.columns)
# print("\n\n\n----------------------------------------------------------------------------------------------------")
# print("Features Extracted & Classified with KNN")

# # ------------------------------------------
# # Redo KNN Classification
# # ------------------------------------------

# # Define the number of folds for cross-validation
# num_folds = 5
# # Initialize a KFold object
# kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# # Initialize a list to store the accuracy scores for each fold
# Accuracy_Scores = []
# # Initialize a list to store the error scores for each fold
# Error_Scores = []
# # Initialize a list to store the sensitivity scores for each fold
# Sensitivity_Scores = []
# # Initialize a list to store the specificity scores for each fold
# Specificity_Scores = []
# # Initialize a list to store the F1 scores for each fold
# F1_Scores = []

# print("----------------------------------------------------------------------------------------------------")
# print("\t\tFold \t Accuracy \t\t Error \t\t Sensitivity \t Specificity \t\t F1 Score")
# print("---------------------------------------------------------------------------------------------------")

# # Iterate through each fold
# for f, (train_index, test_index) in enumerate(kf.split(Features_Only_Round_3)):
#     Training_Features, Testing_Features = Features_Only_Round_3.loc[train_index],Features_Only_Round_3.loc[test_index]
#     Training_Labels, Testing_Labels = Labels_Only_Round_3.loc[train_index],Labels_Only_Round_3.loc[test_index]
    
#     # Class distribution and sampling - START
#     # =============================================================================
    
#     print("-----------------------------------------------")
#     print("Class Distribution")
#     print("-----------------------------------------------")
#     class_distribution = Training_Labels.value_counts()
#     print("Class - no -> " +"{:.3f}".format((class_distribution[0]/Training_Labels.size)*100) + " %")
#     print("Class - yes -> " +"{:.3f}".format((class_distribution[1]/Training_Labels.size)*100) + " %")
    
#     # keep a copy of Training_Features, Training_Labels for future comparsion
#     Training_Features_Original = Training_Features
#     Training_Labels_Original = Training_Labels
#     # Apply SMOTE to balance the dataset
#     # Comment/Uncomment next 2 lines for activating Sampling
#     smote = SMOTE()
#     Training_Features, Training_Labels = smote.fit_resample(Training_Features,Training_Labels)
    
#     print("-----------------------------------------------")
#     print("Class Distribution - After SMOTE Oversampling")
#     print("-----------------------------------------------")
#     class_distribution2 = Training_Labels.value_counts()
#     print("Class - no -> " +"{:.3f}".format((class_distribution2[0]/Training_Labels.size)*100) + " %")
#     print("Class - yes -> " +"{:.3f}".format((class_distribution2[1]/Training_Labels.size)*100) + " %" +"\n\n\n")
    
#     # =============================================================================
#     # Class distribution and sampling - FINISH
#     # =============================================================================
    
#     # Initialize knn classifier
#     KNN_Classifier = KNeighborsClassifier(n_neighbors=113)
    
#     # train the knn classifier
#     KNN_Classifier.fit(Training_Features, Training_Labels)
    
#     # Make predictions
#     Predicted_Labels = KNN_Classifier.predict(Testing_Features)
    
#     # Calculate confusion matrix
#     Confusion_Matrix = confusion_matrix(Testing_Labels, Predicted_Labels)
    
#     # Calculate accuracy for this fold
#     Accuracy = accuracy_score(Testing_Labels, Predicted_Labels)
#     # Append the accuracy score to the list
#     Accuracy_Scores.append(Accuracy)
    
#     # Calculate error for this fold
#     Error = 1 - Accuracy
#     # Append the error score to the list
#     Error_Scores.append(Error)
    
#     # Calculate sensitivity and specificity
#     True_Negatives = Confusion_Matrix[0, 0]
#     False_Positives = Confusion_Matrix[0, 1]
#     False_Negatives = Confusion_Matrix[1, 0]
#     True_Positives = Confusion_Matrix[1, 1]
    
#     Sensitivity = True_Positives / (True_Positives + False_Negatives)
#     # Append the sensitivity score to the list
#     Sensitivity_Scores.append(Sensitivity)
    
#     Specificity = True_Negatives / (True_Negatives + False_Positives)
#     # Append the specificity score to the list
#     Specificity_Scores.append(Specificity)
    
#     # Calculate Precision and Recall from the confusion matrix
#     Precision = True_Positives / (True_Positives + False_Positives)
#     Recall = True_Positives / (True_Positives + False_Negatives)
    
#     # Calculate F1 score
#     f1 = 2 * (Precision * Recall) / (Precision + Recall)
#     # Append the f1 score to the list
#     F1_Scores.append(f1)
    
#     print(" \t\t" + str(f) + " \t\t {:.3f}".format(Accuracy*100) + " \t\t {:.3f}".
#     format(Error*100) + " \t\t {:.3f}".format(Sensitivity*100) + " \t\t {:.3f}".format
#     (Specificity*100) + " \t\t {:.3f}".format(f1*100))
    
#     # Classification Report can be printed/checked for each fold
#     print("Classification Report:")
#     print(classification_report(Testing_Labels, Predicted_Labels))

#     # Printing separatrion line to indicate 5-fold execution completion
#     print("----------------------------------------------------------------------------------------------------")
# # Calculate the average accuracy across all folds
# Average_Accuracy = sum(Accuracy_Scores) / len(Accuracy_Scores)
# # Calculate the average error across all folds
# Average_Error = sum(Error_Scores) / len(Error_Scores)
# # Calculate the average sensitivity across all folds
# Average_Sensitivity = sum(Sensitivity_Scores) / len(Sensitivity_Scores)
# # Calculate the average specificity across all folds
# Average_Specificity = sum(Specificity_Scores) / len(Specificity_Scores)
# # Calculate the average f1 across all folds
# Average_F1 = sum(F1_Scores) / len(F1_Scores)
# # Printing the average performance scores
# print("Average ->" + " \t\t {:.3f}".format(Average_Accuracy*100) + " \t\t {:.3f}".format(Average_Error*100) + " \t\t {:.3f}".format(Average_Sensitivity*100) + " \t\t{:.3f}".format(Average_Specificity*100) + " \t\t {:.3f}".format(Average_F1*100))
# print("----------------------------------------------------------------------------------------------------")

# # ------------------------------------
# # Plot the average performances
# # ------------------------------------

# # Create a new figure and set its size
# plt.figure(figsize=(15, 15))

# plt.subplot(2, 2, 1)
# # Create a bar plot
# plt.bar(['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'], np.array(Accuracy_Scores)*100)
# # Add title
# plt.title('Accuracy over each fold')
# # Add labels
# # plt.xlabel('Folds')
# plt.ylabel('Accuracy (%)')
# # Set y-axis limit
# plt.ylim(0, 100)

# plt.subplot(2, 2, 2)
# # Create a bar plot
# plt.bar(['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'], np.array(Sensitivity_Scores)*100)
# # Add title
# plt.title('Sensitivity over each fold')
# # Add labels
# # plt.xlabel('Folds')
# plt.ylabel('Sensitivity (%)')
# # Set y-axis limit
# plt.ylim(0, 100)

# plt.subplot(2, 2, 3)
# # Create a bar plot
# plt.bar(['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'], np.array(Specificity_Scores)*100)
# # Add title
# plt.title('Specificity over each fold')
# # Add labels
# # plt.xlabel('Folds')
# plt.ylabel('Specificity (%)')
# # Set y-axis limit
# plt.ylim(0, 100)

# plt.subplot(2, 2, 4)
# # Create a bar plot
# plt.bar(['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5'], np.array(F1_Scores)*100)
# # Add title
# plt.title('F1 score over each fold')
# # Add labels
# # plt.xlabel('Folds')
# plt.ylabel('F1 Score (%)')
# # Set y-axis limit
# plt.ylim(0, 100)

# # Show the plot
# plt.show()

