import glob, re, os

files = sorted(glob.glob('reports/analysis/*_*.out'))

# read with utf-16 fallback
def read_file(path):
    for enc in ['utf-8','utf-16','utf-16-le','utf-16-be','latin-1']:
        try:
            with open(path,'r',encoding=enc) as f:
                txt=f.read().replace('\x00','')
                return txt
        except Exception:
            continue
    return ''

pat_dist = re.compile(r'Distance to vertex\s*(\d+)\s*:\s*(.+)')
pat_time = re.compile(r'Execution time:\s*([0-9.eE+-]+)\s*seconds')
pat_neg = re.compile(r'Negative-weight cycle detected|No negative-weight cycles detected', re.IGNORECASE)

# collect per-file data
data = {}
for f in files:
    txt = read_file(f)
    dists = {int(m.group(1)): m.group(2).strip() for m in pat_dist.finditer(txt)}
    time_m = pat_time.search(txt)
    time = float(time_m.group(1)) if time_m else None
    neg_m = pat_neg.search(txt)
    neg = None
    if neg_m:
        t = neg_m.group(0)
        if 'no' in t.lower(): neg = False
        else: neg = True
    data[os.path.normpath(f)] = {'dists': dists, 'time': time, 'neg': neg, 'raw': txt}

# compare to serial baseline
graphs = ['tiny','small','medium','large']
impls = ['serial','openmp','mpi','hybrid','cuda']

lines = []
for g in graphs:
    basef = os.path.normpath(f'reports/analysis/serial_{g}.out')
    base = data.get(basef)
    if not base:
        lines.append(f'{g}: serial output missing')
        continue
    lines.append(f'=== {g} ===')
    lines.append(f'serial: time={base["time"]} neg={base["neg"]} dists={len(base["dists"]) }')
    for impl in impls[1:]:
        fpath = os.path.normpath(f'reports/analysis/{impl}_{g}.out')
        cur = data.get(fpath)
        if not cur:
            lines.append(f'  {impl}: missing')
            continue
        # neg match
        neg_match = (cur['neg'] == base['neg'])
        # compare distances for keys present in both
        common_keys = set(base['dists'].keys()) & set(cur['dists'].keys())
        dist_mismatch = None
        for k in sorted(common_keys):
            if base['dists'][k] != cur['dists'][k]:
                dist_mismatch = (k, base['dists'][k], cur['dists'][k])
                break
        lines.append(f'  {impl}: time={cur["time"]} neg={cur["neg"]} neg_match={neg_match} common_dist_keys={len(common_keys)} dist_mismatch={dist_mismatch}')

print('\n'.join(lines))
