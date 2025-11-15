import os

def print_tree(root, prefix=""):
    print(prefix + os.path.basename(root) + "/")
    for item in sorted(os.listdir(root)):
        path = os.path.join(root, item)
        if os.path.isdir(path):
            print_tree(path, prefix + "    ")
        else:
            print(prefix + "    " + item)

# Set your project root here
project_root = os.path.abspath(".")  # current folder
print_tree(project_root)
