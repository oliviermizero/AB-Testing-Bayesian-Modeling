import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv"
ab_test = pd.read_csv(url)

# Separate data by version (gate_30 and gate_40)
gate_30_data = ab_test[ab_test['version'] == 'gate_30']
gate_40_data = ab_test[ab_test['version'] == 'gate_40']

# Summarize retention rates for 1-day and 7-day retention
retention_1_gate_30 = gate_30_data['retention_1'].mean()
retention_1_gate_40 = gate_40_data['retention_1'].mean()

retention_7_gate_30 = gate_30_data['retention_7'].mean()
retention_7_gate_40 = gate_40_data['retention_7'].mean()

# Convert retention data to arrays for Bayesian modeling
retention_1_gate_30_obs = gate_30_data['retention_1'].astype(int).values
retention_1_gate_40_obs = gate_40_data['retention_1'].astype(int).values

retention_7_gate_30_obs = gate_30_data['retention_7'].astype(int).values
retention_7_gate_40_obs = gate_40_data['retention_7'].astype(int).values

# Bayesian A/B testing for 1-day retention
with pm.Model() as model_1_day:
    p_gate_30 = pm.Beta('p_gate_30', alpha=1, beta=1)
    p_gate_40 = pm.Beta('p_gate_40', alpha=1, beta=1)

    obs_gate_30 = pm.Bernoulli('obs_gate_30', p=p_gate_30, observed=retention_1_gate_30_obs)
    obs_gate_40 = pm.Bernoulli('obs_gate_40', p=p_gate_40, observed=retention_1_gate_40_obs)

    step = pm.Metropolis()
    trace_1_day = pm.sample(2000, tune=1000, step=step, return_inferencedata=True, random_seed=42)

# Bayesian A/B testing for 7-day retention
with pm.Model() as model_7_day:
    p_gate_30 = pm.Beta('p_gate_30', alpha=1, beta=1)
    p_gate_40 = pm.Beta('p_gate_40', alpha=1, beta=1)

    obs_gate_30 = pm.Bernoulli('obs_gate_30', p=p_gate_30, observed=retention_7_gate_30_obs)
    obs_gate_40 = pm.Bernoulli('obs_gate_40', p=p_gate_40, observed=retention_7_gate_40_obs)

    step = pm.Metropolis()
    trace_7_day = pm.sample(2000, tune=1000, step=step, return_inferencedata=True, random_seed=42)

# Summarize and visualize results for both tests
# I got help from Copilot to develop part of this code
az_summary_1_day = az.summary(trace_1_day, var_names=['p_gate_30', 'p_gate_40'])
az_summary_7_day = az.summary(trace_7_day, var_names=['p_gate_30', 'p_gate_40'])

az.plot_posterior(trace_1_day, var_names=['p_gate_30', 'p_gate_40'], hdi_prob=0.95)
plt.title('Posterior Distribution of 1-Day Retention Probabilities')
plt.show()

az.plot_posterior(trace_7_day, var_names=['p_gate_30', 'p_gate_40'], hdi_prob=0.95)
plt.title('Posterior Distribution of 7-Day Retention Probabilities')
plt.show()

# Return the summaries
# I got help from Copilot to develop part of this code
az_summary_1_day, az_summary_7_day