import glob, re
files = sorted(glob.glob('reports/analysis/*_*.out'))
pat = re.compile(r'Distance to vertex\s*(\d+)\s*:\s*(.+)')
for f in files:
    txt=''
    for enc in ['utf-8','utf-16','utf-16-le','utf-16-be','latin-1']:
        try:
            with open(f,'r',encoding=enc) as fh:
                txt=fh.read().replace('\x00','')
                break
        except Exception:
            continue
    found = pat.findall(txt)
    print(f, '->', len(found))
    if len(found)>0:
        print(' sample:', found[:5])
