# deepV
## DeepV Software Description
### 1. Basic Information
Name: DeepV
Version: 1.1
Summary: A CNN-based deep learning tool capable of graphically identifying transcription factor footprints on chromatin from chromatin accessibility data.
License: GPL-3.0
Architecture: x86_64
Related Field: Bioinformatics (Genomic Analysis)
Homepage: https://github.com/example/DeepV
Documentation: https://github.com/example/DeepV/wiki
### 2. Runtime Environment Requirements
2.1 Core Dependencies
Python Version: 3.8
System Libraries/Tools: wget, curl, git, gcc, g++, make, build-essential, libgl1-mesa-glx, libglib2.0-0, libsm6, libxext6, libxrender-dev, libhdf5-dev, pkg-config, tabix, parallel
Python Libraries: pandas, numpy==1.24.3, tensorflow, matplotlib, scipy, scikit-learn, h5py, keras, opencv-python-headless
### 2.2 Basic Environment
Operating System Architecture: x86_64
Working Directory: main directory

### 3. Core Configuration
Run with non-root user (deepvuser) to ensure security.
Required directory structure: /data/step1, /data/step2, /data/logs, /data/final_results, /data/temp_bed_files.
Key files: hg38_chromsize.txt, run_deepv_pipeline.sh (main execution script).
Summary
DeepV v1.1 is a bioinformatics tool for genomic analysis, based on Python 3.8 and CNN deep learning.
It requires specific system and Python dependencies, runs on x86_64 architecture, and uses port 9000.
The tool follows GPL-3.0 license and runs as non-root user for security.

## Usage
Usage: bash run_deepv_pipeline.sh -p <work_dir> -j <max_jobs> -b <bed_file>
Test command: bash run_deepv_pipeline.sh -p /your_path/deepv -j 10 -b /your_path/deepv/test_file/chr22.bed
Options:
  -p, --path      Working directory path (required)
  -j, --jobs      Maximum number of parallel jobs (default: 10)
  -b, --bed       Input BED file path (required)
  -h, --help      Display this help message

Input BED File Format:
  The input BED file should contain 4 columns:
    Column 1: Chromosome name (e.g., chrY)
    Column 2: Fragment midpoint coordinate
    Column 3: Fragment length
    Column 4: Fragment count number
  
  Example format:
    chrY    669491    94    1
    chrY    1169870   71    1
    chrY    1217535   61    1
    chrY    1411077   81    1
    chrY    2353754   67    1
    chrY    2781515   74    1

