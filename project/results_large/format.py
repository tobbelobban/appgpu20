import sys, copy

# order is: 2 4 0 1 3

input = open(sys.argv[1]).read()
numWords = int(sys.argv[2])
extractPos = int(sys.argv[3])

data = [float(tmp.split()[extractPos]) for tmp in input.split('\n')[:-1]]
tmp = copy.deepcopy(data)
data[0] = tmp[2]
data[1] = tmp[4]
data[2] = tmp[0]
data[3] = tmp[1]
data[4] = tmp[3]

string = "["
for i in range(len(data)):
    string += str(data[i])
    if i != len(data)-1:
        string += " "
string += '];'
print(string)
