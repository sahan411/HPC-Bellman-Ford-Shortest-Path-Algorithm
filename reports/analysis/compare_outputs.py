import glob
import re

files = sorted(glob.glob('reports/analysis/*_*.out'))

# read function tries utf-8 then utf-16-le
def read_file(path):
    for enc in ['utf-8', 'utf-16', 'utf-16-le', 'utf-16-be', 'latin-1']:
        try:
            with open(path, 'r', encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ''

# parse distances
pat = re.compile(r'Distance to vertex (\d+): (.+)')

results = {}
for f in files:
    txt = read_file(f)
    d = {}
    for m in pat.finditer(txt):
        idx = int(m.group(1))
        val = m.group(2).strip()
        d[idx] = val
    results[f] = d

# compare each implementation to serial baseline for each graph
graphs = ['tiny','small','medium','large']
impls = ['serial','openmp','mpi','hybrid','cuda']

reports = []
for g in graphs:
    base = results.get(f'reports/analysis/serial_{g}.out', {})
    for impl in impls[1:]:
        cur = results.get(f'reports/analysis/{impl}_{g}.out', {})
        same = True
        # compare keys present in base
        for k in sorted(base.keys()):
            if k not in cur or cur[k] != base[k]:
                same = False
                reports.append(f"{g}: {impl} differs at vertex {k}: serial='{base[k]}', {impl}='{cur.get(k)}'")
                break
        if same:
            reports.append(f"{g}: {impl} matches serial for first {len(base)} vertices")

# print report
print('\n'.join(reports))

# exit code 0
