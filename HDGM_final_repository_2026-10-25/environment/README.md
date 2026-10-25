# Environment evidence
requirements.txt is a dependency list, NOT a historical version lock.
The copied desktop virtual environment points to a missing Python 3.9.12 installation
under a previous user's path. Its interpreter cannot currently start.
The author's Python 3.14 statement therefore cannot be established for all historical
training runs from this directory. Do not substitute packaging-machine versions as
historical experiment versions. Install a PyTorch-compatible Python environment and
record actual installed versions before a full rerun. No virtual environment is bundled.

Reconstruction execution environment (NOT historical training lock): workspace .venv-hdgm, Python3.12.14, NumPy2.5.3, Matplotlib3.11.2. D/E/F executed successfully.
