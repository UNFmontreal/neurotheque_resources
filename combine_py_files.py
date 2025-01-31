import os
import argparse

def combine_py_files(root_dir, output_file):
    """
    Recursively combines all .py files from a root directory into a single output file.
    Includes the file path as a header before each file's content.
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith('.py') and file != os.path.basename(output_file):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as infile:
                                content = infile.read()
                                outfile.write(f"\n\n# {'=' * 50}\n")
                                outfile.write(f"# FILE: {file_path}\n")
                                outfile.write(f"# {'=' * 50}\n\n")
                                outfile.write(content)
                        except Exception as e:
                            print(f"Error reading {file_path}: {str(e)}")
        print(f"Successfully combined files to {output_file}")
    except Exception as e:
        print(f"Error creating output file: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine Python files from a directory into a single file')
    parser.add_argument('--root', default='.', help='Root directory to search from')
    parser.add_argument('--output', default='combined_code.py', help='Output filename')
    args = parser.parse_args()
    
    combine_py_files(args.root, args.output)