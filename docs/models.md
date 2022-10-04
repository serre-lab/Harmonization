# Model zoo

In our experiments, we re-trained a set of models with the harmonization loss proposed in the paper.
You can easily download the weights of each models here:

- **ViT B16 Harmonized**: [serrelab/prj_harmonization/vit_b16_harmonized](https://storage.googleapis.com/serrelab/prj_harmonization/models/vit-b16_harmonized.h5)
- **VGG16 Harmonized**: [serrelab/prj_harmonization/vgg16_harmonized](https://storage.googleapis.com/serrelab/prj_harmonization/models/vgg16_harmonized.h5)
- **ResNet50V2 Harmonized**: [serrelab/prj_harmonization/resnet50v2_harmonized](https://storage.googleapis.com/serrelab/prj_harmonization/models/resnet50v2_harmonized.h5)
- **EfficientNet B0**: [serrelab/prj_harmonization/efficientnet_b0](https://storage.googleapis.com/serrelab/prj_harmonization/models/efficientnetB0_harmonized.h5)


In order to load them easily, we have set up utilities in the github repository.
For example, to load the model lives harmonized:

```python
from harmonization.models import (load_ViT_B16, load_ResNet50,
                                  load_VGG16, load_EfficientNetB0,
                                  preprocess_input)

vit_harmonized = load_ViT_B16()
vgg_harmonized = load_VGG16()
resnet_harmonized = load_ResNet50()
efficient_harmonized = load_EfficientNetB0()

# load images (in [0, 255])
# ...

images = preprocess_input(images)
predictions = vit_harmonized(images)
```
