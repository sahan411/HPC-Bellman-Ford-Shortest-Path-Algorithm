p='reports/analysis/serial_tiny.out'
for enc in ['utf-8','utf-16','utf-16-le','utf-16-be','latin-1']:
    try:
        with open(p,'r',encoding=enc) as f:
            txt=f.read()
            s=txt.replace('\x00','')
            print('===ENC',enc,'===')
            print(repr(s[:500]))
    except Exception as e:
        print('ERR',enc,e)
