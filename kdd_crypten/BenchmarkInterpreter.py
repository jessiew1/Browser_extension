

"""
After running the command `python Benchmark.py > benchmark.results.1`, run the command
`python BenchmarkInterpreter.py` to graph the results.
"""

import matplotlib.pyplot as plt
import re
import math

with open('benchmark.results.1', 'r') as f:
    content = f.readlines()

content = [line for line in content if line != '']

pattern = r'k (\d+) b (\d+\.\d{2}) m (\d+\.\d{2}) mw (\d+) mb (\d+) mr (\d+\.\d{2})'
pattern = re.compile(pattern)

def process(line):
    match = re.match(pattern, line)
    assert match is not None
    
    k = match.expand(r'\1')
    b = match.expand(r'\2')
    ma = match.expand(r'\3')
    mw = match.expand(r'\4')
    mb = match.expand(r'\5')
    mr = match.expand(r'\6')
    
    k = int(k)
    b = float(b)
    ma = float(ma)
    mw = int(mw)
    mb = int(mb)
    mr = float(mr)
    
    return (
        k,
        b,
        mw,
        ma,
        mb,
        # mr,
    )

content = list(map(process, content))
print(content)

for i in range(1, len(content[0])):
    plt.plot(
        list(map(lambda x: x[0], content)),
        list(map(lambda x: x[i], content)),
        label = [
            'N',
            'Sorting Network',
            'Merge Sort Worst',
            'Merge Sort Average',
            'Merge Sort Best',
            # 'Merge Sort Random',
        ][i],
    )
plt.plot(
    list(map(lambda x: x[0], content)),
    list(map(lambda x: x[0] * math.log(x[0]) / math.log(2), content)),
    label = 'n log_2(n)',
)
plt.plot(
    list(map(lambda x: x[0], content)),
    list(map(lambda x: 0.25 * x[0] * (math.log(x[0]) / math.log(2)) ** 2, content)),
    label = '0.25n (log_2(n))^2',
)
    
plt.xlabel('Number of people in voting region')
plt.ylabel('Number of greater than comparisons')
plt.title('Cost of thresholding implementations')
plt.legend(loc = 'upper left')
plt.show()