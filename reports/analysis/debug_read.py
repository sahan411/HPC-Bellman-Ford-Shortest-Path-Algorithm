p='reports/analysis/serial_tiny.out'
with open(p,'rb') as f:
    b=f.read(1000)
print(b[:500])
