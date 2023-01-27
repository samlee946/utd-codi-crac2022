This is the code of our discourse deixis resolution system. 

### Setup

1. Install dependencies   
   ```pip install -r requirements.txt```
2. Set `data_dir` in [experiments.conf](experiments.conf)
3. Run training/inference using your preprocessed data

### Training

```python run.py config_name gpu_id random_seed```

### Batch Evaluation

```python run.py batch gpu_id random_seed```  