import os
os.environ['LD_LIBRARY_PATH']="/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
os.system("python train.py")