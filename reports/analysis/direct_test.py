from importlib import import_module

mods = ['bellman_ford_serial','bellman_ford_openmp','bellman_ford_mpi','bellman_ford_hybrid','bellman_ford_cuda']
for m in mods:
    mod = import_module('bin.'+m)
    func_name = [n for n in dir(mod) if n.startswith('bellman_ford_')][0]
    func = getattr(mod, func_name)
    print('---', m, '---')
    try:
        func('graphs/tiny.txt', 0)
    except SystemExit:
        pass
    except Exception as e:
        print('ERROR calling', m, e)
