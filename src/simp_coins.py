import random

def coinToss(number):
    head_counter = 0
    for i in range(number): # do this 'number' amount of times
         flip = random.randint(0, 1)
         if flip==0: head_counter += 1
    return head_counter

meta_counter = []
for tosses in [100, 1000, 10000]:
    for seed in range(1000):
        random.seed(seed)
        meta_counter.append(coinToss(tosses))
print(meta_counter)