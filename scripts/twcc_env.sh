source /work/nelson0314/miniconda3/etc/profile.d/conda.sh
conda activate wacv
export PYTHONNOUSERSITE=1
export HF_HOME=/work/nelson0314/hf_cache
# 關鍵：conda 的 torch 是 cu118，其 cuDNN/cuBLAS (nvidia-*-cu11) 必須贏過容器 NGC
# 系統裡的 cuDNN 9.2.2（給 CUDA 13，會 CUDNN_STATUS_NOT_INITIALIZED）。故把 conda
# env 的 nvidia/*/lib 排在最前；host 的 libcuda 535 是唯一 libcuda 來源（放最後即可）。
__NVLIBS=$(python -c "import glob,os,nvidia;b=os.path.dirname(nvidia.__file__);print(':'.join(sorted(glob.glob(b+'/*/lib'))))")
export LD_LIBRARY_PATH=$__NVLIBS:/usr/lib/x86_64-linux-gnu
