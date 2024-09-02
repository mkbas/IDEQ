# IDEQ

## Installation

Code is largely built on [T2TCO](https://github.com/Thinklab-SJTU/T2TCO) and [DIFUSCO](https://github.com/Edward-Sun/DIFUSCO).
Environnement installation has to follow T2TCO's environnement specifications as well as the additional cython package for merging the diffusion heatmap results:

```bash
cd utils/cython_merge
python setup.py build_ext --inplace
cd -
```

## Reproducing scripts

Use the following code to reproduce the results of the paper: 

### TSP 500:
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
  --sequential_sampling 1 
  --rewrite_ratio 0.25 
  --norm 
  --rewrite 
  --fp16 
  --do_test_only 
  --n_rep 0 
  --rewrite_steps 3 
  --new_denoise 
  --res_file ...
```

for TSP 500 with seach (N=4) :
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
  --new_denoise 
  --res_file ...
```

### TSP 1000:
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
  --new_denoise 
  --res_file ...
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
  --new_denoise 
  --res_file ...
```
## Checkpoints

IDEQ chekpoints can be downloaded from these links: [TSP500](https://drive.google.com/file/d/1KQMl7-8VglVkfah5hwAnuu0yc0u4LNjg/view?usp=sharing), [TSP1000](https://drive.google.com/file/d/10duH0TW_kl8Or3teJf4zqhAfGJTViYeN/view?usp=sharing)
