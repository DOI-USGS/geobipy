import argparse

Parser = argparse.ArgumentParser(description="update_version", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
Parser.add_argument('version', help='version')
args = Parser.parse_args()

with open("documentation_source/source/conf.py", "r") as f:
        lines = f.readlines()
with open("documentation_source/source/conf.py", "w") as f:
        for line in lines:
            if "version =" in line:
                f.write('version = "{}"\n'.format(args.version))
            elif "release =" in line:
                f.write('release = "{}"\n'.format(args.version))
            else:
                f.write(line)

try:
    with open("pyproject.toml", "r") as f:
        lines = f.readlines()

    with open("pyproject.toml", "w") as f:
        for line in lines:
            if "version =" in line:
                f.write('version = "{}"\n'.format(args.version))
            else:
                f.write(line)
except:
    pass