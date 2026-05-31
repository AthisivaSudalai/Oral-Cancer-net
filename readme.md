On Kaggle, at the top of every notebook:

import sys
import os

# Clone the repo into working directory (first time) or pull updates
REPO_DIR = "/kaggle/working/OralCancerNet"

if not os.path.exists(REPO_DIR):
    !git clone https://github.com/YOUR_USERNAME/OralCancerNet.git {REPO_DIR}
else:
    !git -C {REPO_DIR} pull

# Add to Python path so imports work
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print("Repo ready. sys.path updated.")