## Souce code for A Concept Decomposition Perspective for Interpreting Continuous Prompts

You can try to run our code by following steps below.

#### 1. Download dataset
SST-2: [https://gluebenchmark.com/tasks](https://gluebenchmark.com/tasks)<br>

Then move the data into the dataset/sst2
#### 2. Run P-tuning & CD
For P-tuning:
```python
python3 cli.py --data_dir dataset --model_type bert --model_name_or_path ../bert-large-uncased --task_name sst2 --output_dir ./output/bert-large/sst2 --do_eval --do_train --pet_per_gpu_eval_batch_size 128 --pet_per_gpu_train_batch_size 32 --pet_gradient_accumulation_steps 1 --pet_max_seq_length 256 --pet_max_steps 100 --warmup_steps 10 --pattern_ids 2 --learning_rate 1e-4 --embed_size 1024 --pet_repetitions 1 --dev32_examples 32 --eval_set test --split_examples_evenly --train_mode ptuning --train_examples -1 --seed 1 
 ```
For CD:
```python
python3 cli.py --data_dir dataset --model_type bert --model_name_or_path ../bert-large-uncased --task_name sst2 --output_dir ./output/bert-large/sst2 --do_eval --do_train --pet_per_gpu_eval_batch_size 128 --pet_per_gpu_train_batch_size 128 --pet_gradient_accumulation_steps 1 --pet_max_seq_length 256 --pet_max_steps 100 --warmup_steps 10 --pattern_ids 2 --learning_rate 1e-4 --embed_size 1024 --pet_repetitions 1 --dev32_examples 32 --eval_set test --split_examples_evenly --train_mode concept --train_examples -1 --seed 1  -e 1e7 10 --submodular

```