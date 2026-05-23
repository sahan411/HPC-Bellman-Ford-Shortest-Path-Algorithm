import re, glob
pat_time = re.compile(r'Execution time:\s*([0-9.eE+-]+)\s*seconds')
files = sorted(glob.glob('reports/analysis/bellman_ford_*_*.out'))
res = {}
for f in files:
    name = f.split('\\')[-1]
    txt = open(f, 'r', encoding='utf-8', errors='ignore').read()
    m = pat_time.search(txt)
    t = float(m.group(1)) if m else 0.0
    res[name]=t
print(res)
