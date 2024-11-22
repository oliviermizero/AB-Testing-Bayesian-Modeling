import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

df = pd.read_csv("https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/cookie_cats.csv")

# Separate data by version
observations_1_day_A = df[df['version'] == 'gate_30']['retention_1']
observations_1_day_B = df[df['version'] == 'gate_40']['retention_1']
observations_7_day_A = df[df['version'] == 'gate_30']['retention_7']
observations_7_day_B = df[df['version'] == 'gate_40']['retention_7']

# Define the model for 1-day retention
with pm.Model() as model_1_day:
    p_30 = pm.Uniform('p_30', 0, 1)  # Prior for gate_30 retention probability
    p_40 = pm.Uniform('p_40', 0, 1)  # Prior for gate_40 retention probability

    # Likelihood
    obs_30 = pm.Bernoulli("obs_30", p_30, observed=observations_1_day_A)
    obs_40 = pm.Bernoulli("obs_40", p_40, observed=observations_1_day_B)

    # Sampling
    step = pm.Metropolis()
    trace_1_day = pm.sample(20000, step=step, chains=3, random_seed=RANDOM_SEED)

# Posterior samples for 1-day retention
p_30_samples_1_day = np.concatenate(trace_1_day.posterior.p_30.data[:, 1000:])
p_40_samples_1_day = np.concatenate(trace_1_day.posterior.p_40.data[:, 1000:])

# Plot posterior distributions for 1-day retention
plt.figure(figsize=(12, 8))

# Gate_30 posterior
plt.subplot(211)
plt.xlim(0.4, 0.5)
plt.hist(p_30_samples_1_day, histtype='stepfilled', bins=25, alpha=0.85,
         label="Posterior of $p_{30}$", color="#A60628", density=True)
plt.legend(loc="upper right")
plt.title("Posterior Distribution of $p_{30}$ for 1-Day Retention")

# Gate_40 posterior
plt.subplot(212)
plt.xlim(0.4, 0.5)
plt.hist(p_40_samples_1_day, histtype='stepfilled', bins=25, alpha=0.85,
         label="Posterior of $p_{40}$", color="#467821", density=True)
plt.legend(loc="upper right")
plt.title("Posterior Distribution of $p_{40}$ for 1-Day Retention")
plt.tight_layout()
plt.show()

# Define the model for 7-day retention
with pm.Model() as model_7_day:
    p_30 = pm.Uniform('p_30', 0, 1)  # Prior for gate_30 retention probability
    p_40 = pm.Uniform('p_40', 0, 1)  # Prior for gate_40 retention probability

    # Likelihood
    obs_30 = pm.Bernoulli("obs_30", p_30, observed=observations_7_day_A)
    obs_40 = pm.Bernoulli("obs_40", p_40, observed=observations_7_day_B)

    # Sampling
    step = pm.Metropolis()
    trace_7_day = pm.sample(20000, step=step, chains=3, random_seed=RANDOM_SEED)

# Posterior samples for 7-day retention
p_30_samples_7_day = np.concatenate(trace_7_day.posterior.p_30.data[:, 1000:])
p_40_samples_7_day = np.concatenate(trace_7_day.posterior.p_40.data[:, 1000:])

# Plot posterior distributions for 7-day retention
plt.figure(figsize=(12, 8))

# Gate_30 posterior
plt.subplot(211)
plt.xlim(0.1, 0.2)
plt.hist(p_30_samples_7_day, histtype='stepfilled', bins=25, alpha=0.85,
         label="Posterior of $p_{30}$", color="#A60628", density=True)
plt.legend(loc="upper right")
plt.title("Posterior Distribution of $p_{30}$ for 7-Day Retention")

# Gate_40 posterior
plt.subplot(212)
plt.xlim(0.1, 0.2)
plt.hist(p_40_samples_7_day, histtype='stepfilled', bins=25, alpha=0.85,
         label="Posterior of $p_{40}$", color="#467821", density=True)
plt.legend(loc="upper right")
plt.title("Posterior Distribution of $p_{40}$ for 7-Day Retention")
plt.tight_layout()
plt.show()

# Summarize 1-day retention model
# I got help from Copilot for part of this code
az_summary_1_day = az.summary(trace_1_day, var_names=["p_30", "p_40"], hdi_prob=0.95)
print("1-Day Retention Summary:\n", az_summary_1_day)

# Summarize 7-day retention model
# I got help from Copilot for part of this code
az_summary_7_day = az.summary(trace_7_day, var_names=["p_30", "p_40"], hdi_prob=0.95)
print("\n7-Day Retention Summary:\n", az_summary_7_day)