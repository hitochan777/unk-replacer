import fileinput
from collections import Counter

c = Counter(token for line in fileinput.input() for token in line.strip().split(" "))
for word in c.most_common():
    print(word[0]+" "+str(word[1]))
