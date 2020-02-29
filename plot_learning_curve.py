import re

class Results:
    def __init__(self):
        self.epoch = 0
        self.train_acc = 0
        self.dev_acc = 0
        self.time = 0

def read_results(infile: str):
    """
    read results
    """
    f = open(infile)
    exs = []
    for line in f:
        exs.append(line)
    f.close()
    return exs

def processed():

    results = read_results("results.txt")

    R = []
    number_pattern = re.compile("\d+\.?\d+|\d+")

    for x in results:
        x_parsed = re.split(":|,", x)
        r = Results()
        r.epoch = float(number_pattern.findall(x_parsed[0])[0])
        r.train_acc = float(number_pattern.findall(x_parsed[1])[0])
        r.dev_acc = float(number_pattern.findall(x_parsed[2])[0])
        r.time = float(number_pattern.findall(x_parsed[3])[0])
        R.append(r)

    epoch = []
    train_acc = []
    dev_acc = []
    for r in R:
        epoch.append(r.epoch)
        train_acc.append(r.train_acc)
        dev_acc.append(r.dev_acc)

    return epoch, train_acc, dev_acc

if __name__ == '__main__':
    main()
