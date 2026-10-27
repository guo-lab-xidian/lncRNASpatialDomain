import importlib.util as u
import sys

print("python", sys.version.split()[0], sys.executable)
mods = ['numpy', 'scipy', 'pandas', 'h5py', 'anndata', 'scanpy',
        'sklearn', 'matplotlib', 'pyarrow', 'torch', 'igraph', 'leidenalg']
for m in mods:
    if u.find_spec(m):
        try:
            mod = __import__(m)
            print(f"{m:12} {getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"{m:12} IMPORT_ERROR {e}")
    else:
        print(f"{m:12} MISSING")
