# Bias_contrastive_learning
for debiasing via contrastive learning paper

### 1. clone and download first_baseline.ipyynb
using !git clone [주소.git], cloning our project and running in first_baseline.ipynb (run in Colab) 

### 2. Pretraining 
search where train.sh file is, change !chmod+x [경로], ![경로] based on your dir. 

```python
!chmod +x /content/drive/MyDrive/학부연구생/SimCTG/pretraining/train.sh
!/content/drive/MyDrive/학부연구생/SimCTG/pretraining/train.sh
```
The Args are as follows:
* `--model_name`: further pretraining model, use 'bert-base-uncased' or 'bert-large-uncased' or 'roberta-base' or 'roberta-large' 
* `--train_path`: path of train data, search csv path, use 'prof_test.csv' or 'clean.csv' path.  
* `--dev_path`: path of dev data, use wikitext dev data.
* `--total_steps`: steps you want to pretraining, calculate epoch with batch size info. (effective_batch_size X steps = data X epoch) 
* `--save_every`: use same num with total_steps
* '--learning_rate': find best lr..
* '--margin': for best score, you can use 0.9 (1.0) but need experiments
* '--save_path_prefix': save dir path. Later you can use this path for benchmark inference, eval.

### 3. benchmark text 
search wehere 'MABEL' dir is. 
```python
cd /content/drive/MyDrive/학부연구생/MABEL
```

*if you need other requriements, !pip install according to yout error msg.*

this is for inference 
```python
!python -m benchmark.intrinsic.stereoset.predict --seed 26 \
--model BertForMaskedLM \
--model_name_or_path /content/drive/MyDrive/학부연구생/bert_base_plus_lr14_seed74_st350_wm03/training_step_
```
before run this code, find predict.py file and change json dump path (for inference json file save) 
In predict.py file, change this path every time you want to test model inference. 
```python
    with open(f"{args.persistent_dir}/stereoset/[원하는 파일이름].json", "w") as f:
        json.dump(results, f, indent=2)
```

Inference args. 

* '--model': which type of model you want to test. use "BertForMaskedLM" or "RobertaForMaskedLM"
* '--model_name_or_path': path of your pretrained model dir.

Finally check score, find eval.py and change 'args.predictions.file' path with prediction file path. 
In eval.py, this code. 
```python
    args.predictions_file = f"{args.persistent_dir}/stereoset/[방금 전 inference 저장된 파일 이름].json"
```

check gender, race score both. In eval.py, change this code and test twice (same inference file path) 
for domain in ["gender"] -> ["race"] 
```python
        # domain gender, race 모두 체크. 
        for domain in ["gender"]:
            results["intrasentence"][domain] = self.evaluate(
                self.domain2example["intrasentence"][domain]
```










