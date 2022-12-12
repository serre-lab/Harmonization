In order to evaluate your own model on the benchmark, we have made available two notebooks showing how to do it from tensorflow or pytorch.




<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/230px-Tensorflow_logo.svg.png" width=35>
<sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mp0vxUcIsX1QY-_Byo1LU2IRVcqu7gUl) </sub>


<img src="https://pytorch.org/assets/images/pytorch-logo.png" width=35>
<sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bttp-hVnV_agJGhwdRRW6yUBbf-eImRN) </sub>



Or, you can simply use the api directly as follows:

```python
from harmonization.common import load_clickme_val
from harmonization.evaluation import evaluate_clickme

clickme_dataset = load_clickme_val(batch_size = 128)

scores = evaluate_clickme(model = model, # tensorflow or pytorch model
                          clickme_val_dataset = clickme_dataset,
                          preprocess_inputs=preprocessing_function)
print(scores['alignment_score'])
```

!!! warning
    If you are using a Pytorch model, you need to specify a explainer function (see the pytorch notebook).