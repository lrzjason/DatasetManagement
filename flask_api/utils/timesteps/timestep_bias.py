import numpy as np
import matplotlib.pyplot as plt

# Define a function to generate a random number based on the defined probability distribution
def generate_random_number():
    # Generate x values from 1 to 1000
    x = np.linspace(1, 1000, 1000)
    
    # Calculate the probability distribution function (PDF) using a sine function
    pdf = np.sin(x / 750) + 1
    
    # Normalize the PDF to sum up to 1
    pdf /= np.sum(pdf)
    
    # Use the PDF to generate a random number
    random_number = np.random.choice(x, p=pdf)
    
    return random_number

# random_number = generate_random_number() 
# print(random_number)
# Call the function 1000 times and store the results
random_numbers = [[generate_random_number() for _ in range(1000)] for _ in range(10)]
# random_numbers = [generate_random_number() for _ in range(1000)]

# # 

# Plot the histogram of the generated random numbers
plt.figure(figsize=(10, 6))
# plt.figure(figsize=(10, 1))
# plt.hist(random_numbers, bins=50, density=True, alpha=0.6, color=['blue'])
plt.hist(random_numbers, bins=50, density=True, alpha=1, color=['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
plt.xlabel("Random Number")
plt.ylabel("Probability Density")
plt.title("Random Number Distribution")
plt.grid(True)
plt.show()
