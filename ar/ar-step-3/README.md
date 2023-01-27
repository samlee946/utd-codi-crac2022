This is the code for the third step of our 3-step pipelined coreference resolution system. This is used to remove non-referring/non-mentions from the output of the second-step model. 

### Setup

1. Install dependencies   
   ```pip install -r requirements.txt```
2. Set `data_dir` in [experiments.conf](experiments.conf)
3. Run training/inference using your preprocessed data

### Training

```python run.py config_name gpu_id random_seed```

### Batch Evaluation

```python run.py batch gpu_id random_seed```  