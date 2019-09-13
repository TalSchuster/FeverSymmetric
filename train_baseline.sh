mkdir fever_data
cd fever_data
wget https://www.dropbox.com/s/bdwf46sa2gcuf6j/fever.dev.jsonl
wget https://www.dropbox.com/s/v1a0depfg7jp90f/fever.train.jsonl
cd ..

git clone https://github.com/TalSchuster/pytorch-transformers.git
cd pytorch-transformers
pip install -r requirements.txt
pip install tensorboardX scipy sklearn
python setup.py install

python examples/run_glue.py \
  --task_name fever \
  --do_train \
  --do_eval \
  --do_lower_case \
  --model_type bert \
  --data_dir ../fever_data \
  --model_name_or_path bert-base-uncased \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --save_steps 100000 \
  --output_dir trained_models/baseline/
