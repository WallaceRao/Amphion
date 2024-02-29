exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))
export CUDA_VISIBLE_DEVICES=7
export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir:${PYTHONPATH}
export PYTHONIOENCODING=UTF-8
python bins/svc/http_service.py
