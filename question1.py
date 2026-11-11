import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import statsmodels.api as sm
from adjustText import adjust_text
# --------------------------------------------------------------------------------------------
# ~ Part A ~
# For each predictor, fit a simple linear regression model to predict the response.
# Describe your results. In which of the models is there a statistically significant
# association between the predictor and the response? Create some plots to back up
# your assertions.
# --------------------------------------------------------------------------------------------
# H0 : B1 = 0
# HA : B1 != 0

print("\n Part A")

df = pd.read_csv("BostonHousing.csv")

column_list = df.columns.tolist()

colors = list(mcolors.CSS4_COLORS.keys())

print(column_list)

fig, ax = plt.subplots(3, 5, figsize=(10, 4))
ax[2,4].remove()
ax[2,3].remove()
ax = ax.flatten() 
y = df[column_list[0]]
y_name = column_list[0]
df = df.drop(columns=column_list[0])

column_list = df.columns.tolist()

simple_coefficients = {}

# Make plot for each predictor
for v in range(len(column_list)):
    x_name = column_list[v]
    print(f"The following column, {x_name}", end="")

    # Scatter/Raw Data
    X = df[x_name ].to_numpy()

    #Fit the Ordinary Least Squares (OLS) model
    X_with_constant = sm.add_constant(X)
    model = sm.OLS(y, X_with_constant).fit()

    # Collect p_values
    p_values = model.pvalues


    # Model coefficents/intercept
    intercept = model.params.iloc[0]
    slope = model.params.iloc[1]

    # Model Data
    x_line = df[x_name].to_numpy()
    y_line = slope * x_line + intercept
    simple_coefficients[x_name] = model.params.iloc[1]

    #  Plots (Scatter and Line)
    random_color1 = np.random.rand(3)
    random_color2 = np.random.rand(3)
    ax[v].scatter(X, y, color='blue', label='Data Points', zorder=1, alpha=0.2)
    ax[v].plot(df[x_name], model.predict(X_with_constant), color='red', linewidth=2, label=f'Equation: y = {slope}x + {intercept}')

    
    # 
    ax[v].set_title(f"{y_name} vs. {x_name}")
    ax[v].set_ylabel(y_name) 
    ax[v].set_xlabel(x_name) 

    # Hypothesis Test
    if p_values.iloc[1] < 0.05:
        print("\nDecision: Reject H0. The X variable is statistically significant.\n")
    else:
        print("\nDecision: Fail to reject H0. The X variable is not statistically significant.\n")  

plt.tight_layout()
plt.style.use('seaborn-v0_8-whitegrid')
plt.savefig('HW3_PartA.svg')
plt.show()


# --------------------------------------------------------------------------------------------
# ~ Part B ~
# Fit a multiple regression model to predict the response using all of the predictors.
# Describe your results. For which predictors can we reject the null hypothesis H0: βj =
# 0?
# --------------------------------------------------------------------------------------------
print("\n~ Part B ~")


column_list = df.columns.tolist()
X = df[column_list]

# Create model with all predictors
X_with_constant = sm.add_constant(X)
multiple_model = sm.OLS(y, X_with_constant).fit()


# Collect coefficients
multi_coefficients = multiple_model.params.drop("const")
print(multiple_model.summary())
p_values = multiple_model.pvalues.drop("const")

# Hypothesis testing
for variable, p_value in p_values.items():
    if p_value < 0.05:
        print(f"{variable}: Reject H0. Statistically significant.")
    else:
        print(f"{variable}: Fail to reject H0. Not statistically significant.")

# --------------------------------------------------------------------------------------------
# ~ Part C ~
# How do your results from (a) compare to your results from (b)? Create a plot
# displaying the univariate regression coefficients from (a) on the x-axis, and the
# multiple regression coefficients from (b) on the y-axis. That is, each predictor is
# displayed as a single point in the plot. Its coefficient in a simple linear regression
# model is shown on the x-axis, and its coefficient estimate in the multiple linear
# regression model is shown on the y-axis.
# --------------------------------------------------------------------------------------------
print("\n~ Part C ~")
fig_, ax_ = plt.subplots(figsize=(10, 7))

ax_.scatter(simple_coefficients.values(), multi_coefficients, color='royalblue', marker='o')
x=0

texts = []
for predictor in column_list:
    ax_.annotate(
        predictor,
        (simple_coefficients[predictor],
         multi_coefficients[predictor])
    )
    x+=1
min_val = min(min(simple_coefficients.values()), min(multi_coefficients))
max_val = max(max(simple_coefficients.values()), max(multi_coefficients))

ax_.plot([min_val, max_val], [min_val, max_val],
         'r--', label='y = x')

ax_.set_xlabel("Simple Regression Coefficients (Part A)")
ax_.set_ylabel("Multiple Regression Coefficients (Part B)")
ax_.set_title("Simple vs. Multiple Regression Coefficients")
ax_.axline((0, 0), slope=1, color="red", linestyle="--")
ax_.legend()
ax_.grid(alpha=0.3)
fig_.savefig('HW3_PartC_1.svg')
plt.show()

# Exclude NOX
predictors_zoom = [p for p in simple_coefficients.keys() if p != 'nox']

x_zoom = np.array([simple_coefficients[p] for p in predictors_zoom])
y_zoom = np.array([multi_coefficients[p] for p in predictors_zoom])

# Create figure and axes
fig2, ax2 = plt.subplots(figsize=(10, 7))

# Scatter plot
ax2.scatter(x_zoom, y_zoom)

# Reference line
min_val = min(x_zoom.min(), y_zoom.min())
max_val = max(x_zoom.max(), y_zoom.max())

ax2.plot([min_val, max_val], [min_val, max_val],
        'r--', label='y = x')

# Add predictor labels
for i, name in enumerate(predictors_zoom):
    ax2.annotate(name, (x_zoom[i], y_zoom[i]), fontsize=9)

# Labels and title
ax2.set_xlabel('Univariate Regression Coefficients')
ax2.set_ylabel('Multiple Regression Coefficients')
ax2.set_title('Univariate vs. Multiple Regression (Excluding NOX)')

ax2.legend()
ax2.grid(alpha=0.3)
fig2.savefig("HW3_PartC_2.svg")
plt.show()

# --------------------------------------------------------------------------------------------
# ~ Part D ~
# Is there evidence of non-linear association between any of the predictors and the
# response? To answer this question, for each predictor X, fit a model of the form,
# Y = B0 + B1X + B2X^2 + B3X^3 + e
# --------------------------------------------------------------------------------------------
print("\n~ Part D ~")

for predictor in column_list:
    print(predictor, df[predictor].nunique())

nonlinear_results = {}

for predictor in column_list:

    x = df[predictor] - df[predictor].mean()

    X = pd.DataFrame({
        'X': x,
        'X_squared': x ** 2,
        'X_cubed': x ** 3
    })

    X = sm.add_constant(X)

    model = sm.OLS(y, X).fit()

    nonlinear_results[predictor] = model

    print(f"\n{predictor}")

alpha = 0.05

for predictor, model in nonlinear_results.items():

    test = model.f_test('X_squared = 0, X_cubed = 0')
    p_value = float(test.pvalue)

    print(f"\n{predictor}: p-value = {p_value:.6f}")

    if p_value < alpha:
        print("Reject H0: Evidence of nonlinear association.")
    else:
        print("Fail to reject H0: Insufficient evidence of nonlinear association.")