import os

rootpath = os.path.abspath(os.curdir)
path = os.path.join(rootpath, 'rowdata')

def walk_dir(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            print(os.path.join(name))
        for name in dirs:
            print(os.path.join(name))
            walk_dir(os.path.join(path, name))

a = walk_dir(path)