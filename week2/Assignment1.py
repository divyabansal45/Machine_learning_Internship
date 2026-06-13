import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset load karo
df = pd.read_csv("Dataset 2.csv")

# Q1: First 5 records
print("First 5 records:")
print(df.head())

# Q2: Rows and columns
print("\nShape of dataset:")
print(df.shape)

# Q3: Column names
print("\nColumns:")
print(df.columns)

# Q4: Numerical and categorical features
print("\nNumerical columns:")
print(df.select_dtypes(include=['int64', 'float64']).columns)

print("\nCategorical columns:")
print(df.select_dtypes(include=['object']).columns)

# Q5: Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Q6: Average age
print("\nAverage Age:")
print(df['Age'].mean())

# Q7: Average watch hours per week
print("\nAverage Watch Hours Per Week:")
print(df['WatchHoursPerWeek'].mean())

# Q8: Average monthly spending
print("\nAverage Monthly Spend:")
print(df['MonthlySpend'].mean())

# Q9: Users in each subscription category
print("\nSubscription Counts:")
print(df['SubscriptionType'].value_counts())

# Q10: Percentage of users who renewed subscription
renewed_percent = (df['SubscriptionRenewed'] == 'Yes').mean() * 100
print("\nRenewal Percentage:")
print(f"{renewed_percent:.2f}%")


# Q11: Convert categorical columns to numerical
le = LabelEncoder()

for col in ['Gender', 'SubscriptionType', 'FavoriteGenre', 'SubscriptionRenewed']:
    df[col] = le.fit_transform(df[col])

print("\nEncoded Dataset:")
print(df.head())

# Q12: Define X and y for subscription renewal prediction
X = df.drop('SubscriptionRenewed', axis=1)
y = df['SubscriptionRenewed']

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

# Q13: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Q14: Train Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Q15: Evaluate accuracy
y_pred_dt = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("\nDecision Tree Accuracy:", dt_accuracy)

# Q16: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_dt)
print("\nConfusion Matrix:")
print(cm)

# Q17: Train KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("\nKNN Accuracy:", knn_accuracy)

# Q18: Compare accuracies
print("\nModel Comparison")
print("Decision Tree Accuracy:", dt_accuracy)
print("KNN Accuracy:", knn_accuracy)

# Q19: Linear Regression for Monthly Spend prediction
X_reg = df.drop('MonthlySpend', axis=1)
y_reg = df['MonthlySpend']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train_reg, y_train_reg)

# Q20: Predict spending for a new user
new_user = [1001, 25, 0, 1, 20, 2, 3, 10, 1]
prediction = lr.predict([new_user])

print("\nPredicted Monthly Spend is ", prediction[0])


# week 2 assignment1 submission
# Submitted by Divya Bansal
