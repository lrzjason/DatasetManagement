import numpy as np
import matplotlib.pyplot as plt

# Generate x values from 1 to 1000
total_timestep = 1000
bias = 0.075
x = np.linspace(1, 1000, 1000)

# Calculate the probability distribution function (PDF) using a sine function
pdf = np.sin(x / (total_timestep * (1 - bias) / 2)) + 1

# Normalize the PDF to sum up to 1
pdf /= np.sum(pdf)

# Plot the PDF
plt.plot(x, pdf)
plt.title('Probability Distribution Function')
plt.xlabel('x values')
plt.ylabel('PDF')
plt.show()
