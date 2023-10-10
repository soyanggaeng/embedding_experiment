# embedding_experiment

``` linux
python -m venv .env
source .env/bin/activate
conda deactivate

pip install scipy scikit-learn
pip install torch torchvision
pip install transformers
pip install datasets
pip install gensim
```

## conda env
``` linux
conda create --name myenv

```


## dataset
https://huggingface.co/datasets/lucadiliello/english_wikipedia
``` python
from datasets import load_dataset
dataset = load_dataset("lucadiliello/english_wikipedia")
```

### 참고 
- https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForNextSentencePrediction    
- https://huggingface.co/docs/datasets/v1.11.0/loading_datasets.html
- https://huggingface.co/sgugger/glue-mrpc?text=I+like+you.+I+love+you
- https://huggingface.co/docs/datasets/v1.0.1/loading_metrics.html    
