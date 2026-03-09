import zipfile

# Define the file name
zip_file = "photoz_challenge_data.zip"

# Extract all contents to the current directory
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    # Extracting all the contents of the zip file
    zip_ref.extractall(".")
    print(f"Successfully extracted {zip_file}")
