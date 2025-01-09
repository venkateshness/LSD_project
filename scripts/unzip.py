import zipfile
import argparse
import os

# Define the argument parser
parser = argparse.ArgumentParser(description='Unzip a file to a specified directory.')
parser.add_argument('zip_file', type=str, help='Path to the zip file')
parser.add_argument('extract_to', type=str, help='Directory to extract files to')

# Parse the arguments
args = parser.parse_args()

# Ensure the specified output directory exists
if not os.path.exists(args.extract_to):
    os.makedirs(args.extract_to)

# Unzip the file
try:
    with zipfile.ZipFile(args.zip_file, 'r') as zip_ref:
        zip_ref.extractall(args.extract_to)
    print(f"Extraction completed! Files extracted to {args.extract_to}")
except FileNotFoundError:
    print(f"Error: The file '{args.zip_file}' was not found.")
except zipfile.BadZipFile:
    print(f"Error: The file '{args.zip_file}' is not a valid ZIP file.")
