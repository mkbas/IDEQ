# IDEQ

## Installation

Code is largely built on [T2TCO](https://github.com/Thinklab-SJTU/T2TCO) and [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO).
Environnement installation has to follow T2TCO's environnement specifications. In addition it requires compiling the Cmerge.so library from the C file in the utils folder :

```bash
cd utils
gcc -O3 -march=native -shared -o Cmerge.so -fPIC cmerge.c
cd -
```

## Reproducing scripts

Use the following code to reproduce the results of the paper. This contains the fowlling sub-sections: 
- "data generation" where the code is provided to generate the training and testing Euclidean 2D TSP random instances. TSP linstances and their solutions can be downloaded from the [TSPLIB website](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/)
- "Checkpoint training": where the code is provided to retrain the checkpoint with the updated objective. Pre-trained chekcpoint files are also available for downlad in the next section. 
- Testing" where the code provided allows to reproduce the results 

### Data genearation 

### Checkpoint training 

### Testing 

For the results in the table 1 of the draft paper, the test/validation datasets were 2048 random 2D euclidean TSP instances. For Table 2 it was the tsplib instances.

#### TSP 500:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  
python train.py   
  --task tsp 
  --wandb_logger_name ...         #replace '...' with your chosen wandb name
  --storage_path ./ 
  --validation_split ...          #replace '...' with the path/name to your validation dataset 
  --test_split ...                #replace '...' with the path/name to your test dataset
  --validation_examples 8 
  --sparse_factor 50 
  --inference_schedule cosine 
  --inference_diffusion_steps 20 
  --resume_weight_only 
  --ckpt_path ...                  #replace '...' with the path/name to the model chackpoint (see below to download IDEQ checkpoints)
  --two_opt_iterations 5000 
  --sequential_sampling 1 
  --rewrite_ratio 0.25 
  --norm 
  --rewrite 
  --fp16 
  --do_test_only 
  --n_rep 0 
  --rewrite_steps 3 
  --new_denoise 
```

for TSP 500 with search (N=4) :
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  
python train.py   
  --task tsp 
  --wandb_logger_name ... 
  --storage_path ./ 
  --validation_split ...
  --test_split ...
  --validation_examples 8 
  --sparse_factor 50 
  --inference_schedule cosine 
  --inference_diffusion_steps 20 
  --resume_weight_only 
  --ckpt_path ... 
  --two_opt_iterations 5000 
  --sequential_sampling 4 
  --rewrite_ratio 0.25 
  --norm 
  --rewrite 
  --fp16 
  --do_test_only 
  --n_rep 0 
  --rewrite_steps 3 
```

#### TSP 1000:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3  
python train.py   
  --task tsp 
  --wandb_logger_name ... 
  --storage_path ./ 
  --validation_split ...
  --test_split ...
  --validation_examples 8 
  --sparse_factor 100 
  --inference_schedule cosine 
  --inference_diffusion_steps 20 
  --resume_weight_only 
  --ckpt_path ... 
  --two_opt_iterations 5000 
  --sequential_sampling 1 
  --rewrite_ratio 0.25 
  --norm 
  --rewrite 
  --fp16 
  --do_test_only 
  --n_rep 0 
  --rewrite_steps 3 
```

for TSP 1000 with search (N=4) :
```
CUDA_VISIBLE_DEVICES=0,1,2,3  
python train.py   
  --task tsp 
  --wandb_logger_name ... 
  --storage_path ./ 
  --validation_split ...
  --test_split ...
  --validation_examples 8 
  --sparse_factor 100 
  --inference_schedule cosine 
  --inference_diffusion_steps 20 
  --resume_weight_only 
  --ckpt_path ... 
  --two_opt_iterations 5000 
  --sequential_sampling 4 
  --rewrite_ratio 0.25 
  --norm 
  --rewrite 
  --fp16 
  --do_test_only 
  --n_rep 0 
  --rewrite_steps 3 
```
## Downloading pre-trained checkpoints
IDEQ chekpoints can be downloaded from these links: [TSP500](https://drive.google.com/file/d/1KQMl7-8VglVkfah5hwAnuu0yc0u4LNjg/view?usp=sharing), [TSP1000](https://drive.google.com/file/d/10duH0TW_kl8Or3teJf4zqhAfGJTViYeN/view?usp=sharing)
