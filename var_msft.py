# Vector Autoregression Practice #1 - Gas Sensor

# Importing the libraries
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sb

# Importing the dataset
df = pd.read_csv('msft.csv')
df.plot(kind = 'line')
df.drop(columns = ['Date'], inplace = True)

# Plot each Time Series column to look at trends
sb.set_style('darkgrid')

fig, axes = plt.subplots(nrows  = 3, ncols = 2, dpi = 120)
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color = 'indianred', linewidth = 1)
    ax.set_title(df.columns[i])
    plt.tick_params(labelsize = 6)

    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)

plt.suptitle('Time-Series Variables')
plt.tight_layout()

# Granger's Causality Test to check if TS are causing each other
from statsmodels.tsa.stattools import grangercausalitytests

maxlag = 12
test = 'ssr_chi2test'

def grangers_causation_matrix(ds, var, test = 'ssr_chi2test', verbose = False):
    """
    Apply Granger's Causality test to all variables in the dataset to check if each
    of the time-series are causing each other.

    Null-Hypothesis: Coefficients of past values in the VAR equation are 0, i.e,
    Each T.S does not cause the other.
    """
    df = pd.DataFrame(np.zeros(((len(var)), len(var))), columns = var, index = var)

    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(ds[[r, c]], maxlag = maxlag, verbose = True)
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]

            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            df.loc[r, c] = np.min(p_values)

    return df


grangers_causation_matrix(df, var = df.columns, verbose = True)

# Johanssen's Cointegration Test to check for statistically significant relationships
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def coint(df, alpha = 0.05):
    """
    Cointegration Test to check for long-run statistically significant relationships
    b/ variables
    """

    out = coint_johansen(df, -1, 5)
    d = {'0.9': 0, '0.95': 1, '0.99': 2}

    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]

    def adjust(val, length = 6):
        return str(val).ljust(length)

    # Print Summary
    print('\nName  ::  Test Stat > C(95%)   > Signif\n', '---' * 20)

    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), '::', adjust(round(trace, 2), 9), '>', adjust(cvt, 8), '=>', trace > cvt)

coint(df = df)

# Splitting the into training & testing datasets
nobs = 30
df_train, df_test = df[0:-nobs], df[-nobs:]

# ADF Test to chack for stationarity in variables
from statsmodels.tsa.stattools import adfuller

def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")

# Perform ADF test on each variable
for name, column in df_train.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

df_diff = df_train.diff().dropna()

# Perform ADF test on 1st differenced variables
for name, column in df_diff.iteritems():
    adfuller_test(column, name=column.name)
    print('\n')

# Checking the right order
from statsmodels.tsa.api import VAR

model = VAR(df_diff)
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

x = model.select_order(maxlags=12)
print(x.summary())

# Fitting the VAR model of order 1 to the training data
model_fitted = model.fit(11)
model_fitted.summary()

# Durbin Watson's Statistic to check for correlations in residual errors
from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(model_fitted.resid)

for col, val in zip(df.columns, out):
    print((col), ':', round(val, 2))

# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  #> 4

# Input data for forecasting
forecast_input = df_diff.values[-lag_order:]

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns)

def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col)] = (df_train[col].iloc[-1]-df_train[col].iloc[-2]) + df_fc[str(col)].cumsum()
        # Roll back 1st Diff
        df_fc[str(col)+'_forecast'] = df_train[col].iloc[-1] + df_fc[str(col)].cumsum()
    return df_fc

results = invert_transformation(df_train, forecast, second_diff = False)

forecasted_result = np.split(results, [6], axis = 1)[1]

# Actual vs Predicted Plot
fig, axes = plt.subplots(nrows = int(len(df.columns)/2), ncols = 2, dpi = 150, figsize = (8,8))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):

    results[col+'_forecast'].plot(legend = True, ax = ax).autoscale(axis = 'x',tight = True)
    df_test[col][-nobs:].plot(legend = True, ax = ax)
    ax.legend(loc = 'upper center', prop = {'size': 6})

    ax.set_title(col + ": Forecast vs Actuals")

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize = 6)

plt.tight_layout();

# Metrics
from sklearn.metrics import mean_squared_error
from math import sqrt

print('\nRMSE: ', sqrt(mean_squared_error(df_test['Open'], forecasted_result['Open_forecast'])))
