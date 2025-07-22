# UnLoG
Bias mitigation learning code for the paper [UnLoG: A Unified Framework for Multi-Scale Bias Mitigation in Token
Associations and Semantic Interpretations]

### 1. clone and download first_baseline.ipyynb
using !git clone [our_repository.git], cloning our project.

### 2. Pretraining 
Search where train.sh file is, change !chmod+x [shell_file_path], ![shell_file_path] based on your dir. 
For example, 
```python
!chmod +x /content/drive/MyDrive/SimCTG/pretraining/train.sh
!/content/drive/MyDrive/SimCTG/pretraining/train.sh
```
The Args are as follows:
* `--model_name`: backbone model you want to train on, use 'bert-base-uncased', 'bert-large-uncased','roberta-base', 'distilbert-base-uncased', and 'google/electra-base-generator' 
* `--train_path`: path of train data, search csv path, use 'prof_test.csv' or 'clean.csv' path.  
* `--dev_path`: path of dev data, use wikitext dev data.
* `--total_steps`: steps you want to train, calculate epoch with batch size info. (effective_batch_size X steps = data X epoch) 
* `--save_every`: use same num with total_steps
* '--learning_rate': use learning rate in the original paper 
* '--margin': for best score, you can use 0.9 (1.0) but need experiments
* '--save_path_prefix': save dir path. After the training, you can use this path for benchmark inference, evaluations. 

### 3. benchmark text 
search wehere 'MABEL' dir is. For example, 
```python
cd /content/drive/MyDrive//MABEL
```

*if you need other requriements, !pip install according to yout error msg.*

This is the code for inference. 
```python
!python -m benchmark.intrinsic.stereoset.predict --seed 26 \
--model BertForMaskedLM \
--model_name_or_path /content/drive/MyDrive/[save_path_prefix] 
```
Before run this code, find predict.py file and change json dump path (for inference json file save) 
In predict.py file, change this path every time you want to test model inference. 
```python
    with open(f"{args.persistent_dir}/stereoset/[inference_file_name_you_want].json", "w") as f:
        json.dump(results, f, indent=2)
```

Inference args. 

* '--model': which type of model you want to test.
* '--model_name_or_path': path of your pretrained model dir.

Finally check score, find eval.py and change 'args.predictions.file' path with prediction file path. 
In eval.py, this code. 
```python
    args.predictions_file = f"{args.persistent_dir}/stereoset/[inference_file_name_you_want].json"
```

Check gender, race, and other categories. In eval.py, change this code and test twice (same inference file path) 
for domain in ["gender"] -> ["race"] 
```python
        for domain in ["gender"]:
            results["intrasentence"][domain] = self.evaluate(
                self.domain2example["intrasentence"][domain]
```










