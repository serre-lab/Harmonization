In order to train your own harmonized model, we have made available a way to simply load the ClickMe training set, as well as the harmonization loss we have used in the paper.

## Loading ClickMe training set

First, you will need to load the training dataset:

```python
from harmonization.common import load_clickme_train

clickme_ds = load_clickme_train(batch_size = 128)

for images, heatmaps, labels in clickme_ds:
    print(images.shape) # (128, 224, 224, 3)
    print(heatmaps.shape) # (128, 224, 224, 1)
    print(labels.shape) # (128, 1000)

```

Note that, if you already have the shards locally, you can also load the dataset using the `load_clickme` function:


```python
from harmonization.common import load_clickme

clickme_ds = load_clickme_train(shards_paths = ['dataset/train_clickme_0',
                                                'dataset/train_clickme_1'
                                                ...
                                               ], batch_size = 128)
```

## Using the Harmonization loss

Now that we know how to load the training set, we just need the harmonization loss:

```python
def harmonizer_loss(model, images, tokens, labels, true_heatmaps,
                    cross_entropy = tf.keras.losses.CategoricalCrossentropy(),
                    lambda_weights=1e-5, lambda_harmonization=1.0):
                    ...
```

To use the loss, simply call the function with your model, the images / labels and heatmaps for ClickMe:

```python
from harmonization.training import harmonizer_loss

... # loading dataset


for images, heatmaps, labels in clickme_ds:
    tokens = tf.ones(len(images)) # tokens are flags to indicate if the image have an associated heatmap
    loss = harmonizer_loss(model, images, tokens, labels, heatmaps)

```

For example, if we decide to mix the ClickMe dataset with ImageNet, we may not have heatmaps for each images, in that case we can use the `tokens` flag parameters to designate when an heatmaps is provided (`1` means heatmaps provided).

