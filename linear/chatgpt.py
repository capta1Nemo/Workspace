import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_coins = 1000
num_flips = 10
num_experiments = 100000

# Arrays to store the fraction of heads for each type of coin
fraction_heads_c1 = np.zeros(num_experiments)
fraction_heads_crand = np.zeros(num_experiments)
fraction_heads_cmin = np.zeros(num_experiments)

# Run the simulation
for i in range(num_experiments):
    # Simulate flipping each coin 10 times
    flips = np.random.binomial(1, 0.5, (num_coins, num_flips))
    num_heads = np.sum(flips, axis=1)
    fraction_heads = num_heads / num_flips
    
    # c1 is the first coin
    fraction_heads_c1[i] = fraction_heads[0]
    
    # crand is a randomly chosen coin
    crand = np.random.randint(num_coins)
    fraction_heads_crand[i] = fraction_heads[crand]
    
    # cmin is the coin with the minimum frequency of heads
    cmin = np.argmin(num_heads)
    fraction_heads_cmin[i] = fraction_heads[cmin]

# Plot the histograms
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(fraction_heads_c1, bins=np.linspace(0, 1, 11), edgecolor='black')
plt.title("Distribution of fraction of heads (c1)")
plt.xlabel("Fraction of heads")
plt.ylabel("Frequency")

plt.subplot(1, 3, 2)
plt.hist(fraction_heads_crand, bins=np.linspace(0, 1, 11), edgecolor='black')
plt.title("Distribution of fraction of heads (crand)")
plt.xlabel("Fraction of heads")
plt.ylabel("Frequency")

plt.subplot(1, 3, 3)
plt.hist(fraction_heads_cmin, bins=np.linspace(0, 1, 11), edgecolor='black')
plt.title("Distribution of fraction of heads (cmin)")
plt.xlabel("Fraction of heads")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
