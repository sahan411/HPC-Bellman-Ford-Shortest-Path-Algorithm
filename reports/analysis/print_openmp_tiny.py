from pathlib import Path
p=Path('reports/analysis/openmp_tiny.out')
for enc in ['utf-8','utf-16','utf-16-le','latin-1']:
    try:
        txt=p.read_text(encoding=enc).replace('\x00','')
        print('ENC',enc)
        print(txt)
        break
    except Exception as e:
        print('ERR',enc,e)
