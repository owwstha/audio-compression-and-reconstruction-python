# %%
import matplotlib.pyplot as plt
import numpy as np

rl_model = [0, 1.86, 4.4, 6.06, 8.26]
duty_obs_model = [10.31, 24.03, 49.20, 65.33, 100]
duty_calc_model = [10.29, 27.66, 49.5, 72, 100]

rl_actual = [0, 2.5, 5.2, 5.43, 8.43]
duty_obs_actual = [7.26, 28.44, 48.95, 65, 100]
duty_calc_actual = [10.58, 32.04, 55, 74, 100]

plt.figure(figsize=(12, 7))

plt.plot(rl_model, duty_calc_model, color='blue', linestyle='-', marker='', label='Model recon')
plt.plot(rl_model, duty_obs_model, color='blue', linestyle='--', marker='o', label='Model recon')

plt.plot(rl_actual, duty_calc_actual, color='orangered', linestyle='-', marker='', label='Actual recon')
plt.plot(rl_actual, duty_obs_actual, color='orangered', linestyle='--', marker='x', label='Actual recon')

plt.title('Comparison of Model vs. Actual Circuit Duty Cycle', fontsize=16)
plt.xlabel('Load Resistance (Rl) in kΩ', fontsize=12)
plt.ylabel('Duty Cycle (%)', fontsize=12)

plt.legend()

plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.xlim(left=-0.5, right=max(rl_model + rl_actual) + 0.5)
plt.ylim(bottom=0, top=105)

plt.show()
