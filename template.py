import os
from pathlib import Path

# Define the project's root directory
project_name = "src"

# List of files and folders to create
list_of_files = [
    ".github/workflows/.gitkeep",
    f"{project_name}/__init__.py",
    f"{project_name}/data_preprocessing.py",
    f"{project_name}/model_training.py",
    f"{project_name}/api.py",
    f"{project_name}/utils.py",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "models/.gitkeep",
    "notebooks/01_data_exploration_and_baselining.ipynb",
    "tests/__init__.py",
    "tests/test_api.py",
    "requirements.txt",
    "Dockerfile",
    "README.md",
    "demo.py"
]

# Loop through the list to create the files and directories
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # create an empty file
    else:
        print(f"File is already exists: {filepath}")
