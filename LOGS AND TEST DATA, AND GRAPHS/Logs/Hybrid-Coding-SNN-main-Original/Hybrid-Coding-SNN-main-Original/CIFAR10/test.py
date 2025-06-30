import sys
from random import random
import logging
sys.stdout = open('log.txt', 'w')



print(f"a{random()}")
print(f"a{random()}")
print(f"a{random()}")
sys.stdout.close()
sys.stdout = sys.__stdout__
print('end')