import random
import numpy as np
heads=1
tails=0
N=100000
v1=np.zeros(N)
vrand=np.zeros(N)
vmin=np.zeros(N)
for i in range(N):
    flips = np.random.binomial(1, 0.5, (1000, 10))
    num_heads = np.sum(flips, axis=1)
    fraction_heads=num_heads/10
    # v1 is the first coin
    v1[i] = fraction_heads[0]
    
    # vrand is a randomly chosen coin
    crand = np.random.randint(1000)
    vrand[i] = fraction_heads[crand]
    
    # vmin is the coin with the minimum frequency of heads
    cmin = np.argmin(num_heads)
    vmin[i] = fraction_heads[cmin]

print(np.mean(v1),np.mean(vrand),np.mean(vmin))
