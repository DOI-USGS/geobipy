import argparse

Parser = argparse.ArgumentParser(description="update_version", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
Parser.add_argument('version', help='version')
args = Parser.parse_args()

with open("setup.py", "r") as f:
    lines = f.readlines()

with open("setup.py", "w") as f:
    for line in lines:
        if "__version__ =" in line:
            f.write("__version__ = '{}'\n".format(args.version))
        else:
            f.write(line)