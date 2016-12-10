class UnionFind:
    def __init__(self, size):
        # A negative value denotes representative of set and the absolute value denotes the size of set
        # A positive value denotes next element.
        self.table = [-1 for _ in range(size)]

    # Get representative of a set
    def find(self, x):
        while self.table[x] >= 0:
            x = self.table[x]

        return x

    def union(self, x, y):
        s1 = self.find(x)
        s2 = self.find(y)
        if s1 != s2:
            if self.table[s1] >= self.table[s2]:
                # The smaller has more elements
                self.table[s1] += self.table[s2]
                self.table[s2] = s1
            else:
                self.table[s2] += self.table[s1]
                self.table[s1] = s2
            return True
        return False

    def get_groups(self):
        groups = []
        for i in range(len(self.table)):
            groups.append(self.find(i))

        return groups
