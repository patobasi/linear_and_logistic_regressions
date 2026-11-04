import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as mcolors
import statsmodels.api as sm


df = pd.read_csv("BostonHousing.csv")

column_list = df.columns.tolist()

colors = list(mcolors.CSS4_COLORS.keys())

print(column_list)

fig, ax = plt.subplots(3, 4, figsize=(10, 4))

ax = ax.flatten() 
y = df[column_list[0]]
y_name = column_list[0]
df = df.drop(columns=column_list[0])

column_list = df.columns.tolist()
for v in range(len(column_list)-1):
    print(f"The following column, {column_list[v]}", end="")

    # Scatter/Raw Data
    X = df[column_list[v]].to_numpy()

    #Fit the Ordinary Least Squares (OLS) model
    X_with_constant = sm.add_constant(X)
    model = sm.OLS(y, X_with_constant).fit()

    # Collect p_values
    p_values = model.pvalues


    # Model coefficents/intercept
    intercept = model.params.iloc[0]
    slope = model.params.iloc[1]

    # Model Data
    x_line = df[column_list[v]].to_numpy()
    y_line = slope * x_line + intercept

    #  Plots (Scatter and Line)
    ax[v].scatter(X, y, color='blue', label='Data Points', zorder=1, alpha=0.2)
    ax[v].plot(x_line, y_line, color='red', linewidth=2, label=f'Equation: y = {slope}x + {intercept}')

    
    # 
    ax[v].set_title(column_list[v])
    ax[v].set_ylabel(column_list[0]) 

    if p_values.iloc[1] < 0.05:
        print("\nDecision: Reject H0. The X variable is statistically significant.\n")
    else:
        print("\nDecision: Fail to reject H0. The X variable is not statistically significant.\n")  

plt.tight_layout()
plt.show()