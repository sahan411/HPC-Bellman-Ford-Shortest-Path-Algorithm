import subprocess, sys, os
impls = ['bellman_ford_serial.py','bellman_ford_openmp.py','bellman_ford_mpi.py','bellman_ford_hybrid.py','bellman_ford_cuda.py']
graphs = {'tiny':'graphs/tiny.txt','small':'graphs/small.txt','medium':'graphs/medium.txt','large':'graphs/large.txt'}
outdir = 'reports/analysis'
if not os.path.isdir(outdir): os.makedirs(outdir)
python = sys.executable
for impl in impls:
    name = impl.replace('.py','')
    for g, path in graphs.items():
        cmd = [python, os.path.join('bin', impl), path, '0']
        print('RUN:', ' '.join(cmd))
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out = p.stdout
            if p.stderr:
                out += '\nSTDERR:\n'+p.stderr
        except Exception as e:
            out = f'ERROR running: {e}'
        # write utf-8
        of = os.path.join(outdir, f'{name}_{g}.out')
        with open(of, 'w', encoding='utf-8') as f:
            f.write(out)
print('Done')
