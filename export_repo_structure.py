import os
import ast

def get_function_names(file_path):
    """Extract top-level function names from a Python file."""
    function_names = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
    except (SyntaxError, UnicodeDecodeError, Exception):
        pass  # Skip unreadable or non-Python files
    return function_names

def generate_repo_structure(start_path, output_file="repo_structure.txt", exclude_dirs=None):
    """Generate a directory tree with function names in Python files."""
    if exclude_dirs is None:
        exclude_dirs = {".git", "__pycache__", "venv", "env", "node_modules"}

    structure = []
    for root, dirs, files in os.walk(start_path, topdown=True):
        # Exclude unwanted directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        level = root.replace(start_path, "").count(os.sep)
        indent = " " * 4 * level
        structure.append(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                functions = get_function_names(file_path)
                func_list = f" ({', '.join(functions)})" if functions else ""
                structure.append(f"{subindent}{file}{func_list}")
            else:
                structure.append(f"{subindent}{file}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(structure))

if __name__ == "__main__":
    repo_root = os.getcwd()  # Run from the repository root
    generate_repo_structure(repo_root)