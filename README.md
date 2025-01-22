# IDEQ

## Installation

Code is largely built on [T2TCO](https://github.com/Thinklab-SJTU/T2TCO) and [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO).
Environnement installation has to follow T2TCO environment specifications. In addition it requires compiling the Cmerge.so library from the C file in the utils folder:

```bash
cd utils
gcc -O3 -march=native -shared -o Cmerge.so -fPIC cmerge.c
cd -
```

## Reproducing scripts

Use the following code to reproduce the results of the paper. This contains the following sub-sections: 
- "data generation" where contains the code to generate the training and testing Euclidean 2D TSP random instances. TSP instances and their solutions can be downloaded from the [TSPLIB website](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/)
- "Checkpoint training" where contains the code to retrain the checkpoint with the updated objective. Pre-trained checkpoint files are also available for download in the next section. 
- "Testing" contains the code to reproduce the results.

### Data genearation 

To generate the datasets used in our experiments run the following script: 
```bash
cd data
python data_gen.py
cd -
```
Files will be created in the 'data' folder with the names as used in the reproducing scripts below.
Random seeds are fixed (one for the training dataset generation and one for the testing dataset generation), they are the same as we used in our experiments.

### Checkpoint training 

#### IDEQ TSP-500 training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py 
  --task tsp 
  --wandb_logger_name ideq_github                      #update with your chosen wandb name
  --diffusion_type categorical 
  --do_train 
  --learning_rate 0.0002 
  --weight_decay 0.0001 
  --lr_scheduler cosine-decay 
  --storage_path /path/to/IDEQ                         #update with your path to IDEQ
  --training_split data/tsp500_train_concorde.txt      #update with your path the training dataset
  --validation_split data/tsp500_test_concorde.txt     #update with your path the validation dataset
  --test_split data/tsp500_test_concorde.txt           #update with your path the test dataset
  --sparse_factor 50 
  --batch_size 4 
  --num_epochs 50 
  --validation_examples 8 
  --inference_schedule cosine 
  --inference_diffusion_steps 50 
  --rev2opt 
  --use_activation_checkpoint 
  --ckpt_path /path/to/DIFUSCO_TSP-100/checkpoint      #update with your path the DIFUSCO TSP-100 categorical checkpoint 
  --resume_weight_only
```

#### IDEQ TSP-1000 training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py 
  --task tsp 
  --wandb_logger_name ideq_github                       #update with your chosen wandb name
  --diffusion_type categorical 
  --do_train 
  --learning_rate 0.0002 
  --weight_decay 0.0001 
  --lr_scheduler cosine-decay 
  --storage_path /path/to/IDEQ                          #update with your path to IDEQ
  --training_split data/tsp1000_train_concorde.txt      #update with your path the training dataset
  --validation_split data/tsp1000_test_concorde.txt     #update with your path the validation dataset
  --test_split data/tsp1000_test_concorde.txt           #update with your path the test dataset
  --sparse_factor 50 
  --batch_size 8 
  --num_epochs 50 
  --validation_examples 8 
  --inference_schedule cosine 
  --inference_diffusion_steps 50 
  --rev2opt 
  --use_activation_checkpoint 
  --ckpt_path /path/to/IDEQ_TSP-500/checkpoint           #update with your path the IDEQ_TSP-500 checkpoint 
  --resume_weight_only
```

### Testing 

For the results in the table 1 of the submitted paper, the test/validation datasets were 2048 random 2D euclidean TSP instances. For Table 2 it was the tsplib instances.

#### TSP 500:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  
python test.py   
  --task tsp 
  --wandb_logger_name ...                              #replace '...' with your chosen wandb name
  --storage_path /path/to/IDEQ                         #update with your path to IDEQ
  --validation_split data/tsp500_test_concorde.txt     #update with your path the validation dataset
  --test_split data/tsp500_test_concorde.txt           #update with your path the test dataset
  --validation_examples 8 
  --sparse_factor 50 
  --inference_schedule cosine 
  --inference_diffusion_steps 20 
  --resume_weight_only 
  --ckpt_path ...                                      #replace '...' with the path/name to the model checkpoint (see below to download IDEQ checkpoints)
  --two_opt_iterations 5000 
  --sequential_sampling 1 
  --rewrite_ratio 0.25 
  --norm 
  --rewrite 
  --fp16 
  --do_test_only 
  --rewrite_steps 3 
  --new_denoise 
```

for TSP 500 with search (N=4) :
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  
python test.py   
  --task tsp 
  --wandb_logger_name ...                              #replace '...' with your chosen wandb name
  --storage_path /path/to/IDEQ                         #update with your path to IDEQ
  --validation_split data/tsp500_test_concorde.txt     #update with your path the validation dataset
  --test_split data/tsp500_test_concorde.txt           #update with your path the test dataset
  --validation_examples 8 
  --sparse_factor 50 
  --inference_schedule cosine 
  --inference_diffusion_steps 20 
  --resume_weight_only 
  --ckpt_path ...                                      #replace '...' with the path/name to the model checkpoint (see below to download IDEQ checkpoints)
  --two_opt_iterations 5000 
  --sequential_sampling 4 
  --rewrite_ratio 0.25 
  --norm 
  --rewrite 
  --fp16 
  --do_test_only 
  --rewrite_steps 3 
```

#### TSP 1000:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  
python test.py   
  --task tsp 
  --wandb_logger_name ...                              #replace '...' with your chosen wandb name
  --storage_path /path/to/IDEQ                         #update with your path to IDEQ
  --validation_split data/tsp1000_test_concorde.txt    #update with your path the validation dataset
  --test_split data/tsp1000_test_concorde.txt          #update with your path the test dataset
  --validation_examples 8 
  --sparse_factor 50 
  --inference_schedule cosine 
  --inference_diffusion_steps 20 
  --resume_weight_only 
  --ckpt_path ...                                      #replace '...' with the path/name to the model checkpoint (see below to download IDEQ checkpoints)
  --two_opt_iterations 5000 
  --sequential_sampling 1 
  --rewrite_ratio 0.25 
  --norm 
  --rewrite 
  --fp16 
  --do_test_only 
  --rewrite_steps 3 
```

for TSP 1000 with search (N=4) :
```
CUDA_VISIBLE_DEVICES=0,1,2,3  
python test.py   
  --task tsp 
  --wandb_logger_name ...                              #replace '...' with your chosen wandb name
  --storage_path /path/to/IDEQ                         #update with your path to IDEQ
  --validation_split data/tsp1000_test_concorde.txt    #update with your path the validation dataset
  --test_split data/tsp1000_test_concorde.txt          #update with your path the test dataset
  --validation_examples 8 
  --sparse_factor 50 
  --inference_schedule cosine 
  --inference_diffusion_steps 20 
  --resume_weight_only 
  --ckpt_path ...                                      #replace '...' with the path/name to the model checkpoint (see below to download IDEQ checkpoints)
  --two_opt_iterations 5000 
  --sequential_sampling 4 
  --rewrite_ratio 0.25 
  --norm 
  --rewrite 
  --fp16 
  --do_test_only 
  --rewrite_steps 3 
```
## Downloading pre-trained checkpoints
IDEQ chekpoints can be downloaded from these links: [IDEQ_TSP-500](https://drive.google.com/file/d/1KQMl7-8VglVkfah5hwAnuu0yc0u4LNjg/view?usp=sharing), [IDEQ_TSP-1000](https://drive.google.com/file/d/10duH0TW_kl8Or3teJf4zqhAfGJTViYeN/view?usp=sharing)

The DIFUSCO TSP-100 categorical checkpoint that was used to train the IDEQ TQP-500 checkpoint can be download from it's [original publication](https://drive.google.com/file/d/1G2nxIC_qfAswk9TstMFBOCpLQL4rDhKa/view?usp=drive_link)
