import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import seaborn as sns

"""
a) Produce some numerical and graphical summaries of the Weekly data. Do there
appear to be any patterns?
b) Use the full data set to perform a logistic regression with Direction as the response
and the five lag variables plus Volume as predictors. Use the summary function to
print the results. Do any of the predictors appear to be statistically significant? If so,
which ones?
c) Compute the confusion matrix and overall fraction of correct predictions. Explain
what the confusion matrix is telling you about the types of mistakes made by logistic
regression.
d) Now fit the logistic regression model using a training data period from 1990 to 2008,
with Lag2 as the only predictor. Compute the confusion matrix and the overall
fraction of correct predictions for the held-out data (that is, the data from 2009 and
2010)."""

print("\n Part A")

df = pd.read_csv("Smarket.csv")

# Basic numerical summary
print(df.describe())

# Counts of Up and Down
print(df['Direction'].value_counts())

# Correlations between numerical variables
print(df.corr(numeric_only=True))

# Volume over time
# Find the maximum trading volume for each year
yearly_max = df.groupby('Year')['Volume'].max()

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Original scatter plot
ax.scatter(df['Year'], df['Volume'], alpha=0.5, label='Weekly Volume')

# Overlay line connecting the yearly maximum volumes
ax.plot(yearly_max.index, yearly_max.values,
        color='red', marker='o', linewidth=2,
        label='Maximum Volume Each Year')

# Labels and title
ax.set_xlabel('Year')
ax.set_ylabel('Volume')
ax.set_title('Weekly Trading Volume Over Time')
ax.legend()
ax.grid(alpha=0.3)
plt.savefig("HW3_Q2_PartA.svg")
plt.show()



column_list = df.columns.tolist()

df['Direction'] = df['Direction'].map({'Up': 1, 'Down': 0})

X = df[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume']]
y = df['Direction']

X = sm.add_constant(X)

model = sm.Logit(y, X).fit()

print(model.summary())

alpha = 0.05
p_values = model.pvalues.drop("const")

# Hypothesis testing
for variable, p_value in p_values.items():
    if p_value < alpha:
        print(f"{variable}: Reject H0. Statistically significant.")
    else:
        print(f"{variable}: Fail to reject H0. Not statistically significant.")


# Predicted probabilities
predicted_probabilities = model.predict(X)

# Convert probabilities into predicted directions
predicted_direction = (predicted_probabilities >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(y, predicted_direction, labels=[0, 1])
print(cm)
print((cm[0, 0] + cm[1, 1])/sum(sum(row) for row in cm))
class_names = ['Negative', 'Positive']
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)

plt.title('Confusion Matrix')

plt.savefig("HW3_CM.svg")

plt.show()

