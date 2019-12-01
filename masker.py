#Reading text file


import argparse
import sys
import random

# try:

f = open(sys.argv[1], 'r')
text_file = f.read()
words = text_file.split()


masked_percentage = int(sys.argv[2])

masked_number = int((masked_percentage/100)*len(words))
mask = '[MASK]'



while masked_number>0:
    masked = random.choice(words)
    loc = words.index(masked)
    words[loc] = mask
    masked_number -= 1

masked = " ".join(words)

print(text_file)
print(masked)
# except:
#     print("Enter correct arguments")
