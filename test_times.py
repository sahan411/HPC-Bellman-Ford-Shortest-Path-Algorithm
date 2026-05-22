import requests
import json

algorithms = ['serial', 'openmp', 'mpi', 'hybrid', 'cuda']
results = {}

for alg in algorithms:
    payload = {
        'algorithm': alg,
        'graph_file': 'tiny.txt',
        'source': 0,
        'threads': 4,
        'processes': 4
    }
    
    r = requests.post('http://127.0.0.1:5000/api/run', json=payload)
    data = r.json()
    
    if data['success']:
        results[alg] = float(data['time'])
        print(f"{alg.upper():10} : {data['time']} seconds")
    else:
        print(f"{alg.upper():10} : FAILED")

print("\n" + "="*50)
print("EXECUTION TIME COMPARISON (tiny.txt, 5V 8E):")
print("="*50)
for alg in algorithms:
    if alg in results:
        print(f"{alg.upper():10} : {results[alg]:.6f} seconds")
