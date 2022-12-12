
# Harmonizing the object recognition strategies of deep neural networks with humans

<div>
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.6, 3.7, 3.8-efefef">
    </a>
    <a href="https://github.com/serre-lab/Harmonization/actions/workflows/python-lint.yml">
        <img alt="PyLint" src="https://github.com/serre-lab/Harmonization/actions/workflows/python-lint.yml/badge.svg">
    </a>
    <a href="https://github.com/serre-lab/Harmonization/actions/workflows/python-tests.yml">
        <img alt="Tox" src="https://github.com/serre-lab/Harmonization/actions/workflows/python-tests.yml/badge.svg">
    </a>
    <a href="https://github.com/serre-lab/Harmonization/actions/workflows/python-pip.yml">
        <img alt="Pypi" src="https://github.com/serre-lab/Harmonization/actions/workflows/python-pip.yml/badge.svg">
    </a>
    <a href="https://pepy.tech/project/harmonization">
        <img alt="Pepy" src="https://pepy.tech/badge/harmonization">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
</div>

<br>

_Thomas Fel*, Ivan Felipe Rodriguez*, Drew Linsley*, Thomas Serre_

<p align="center">
<a href="https://arxiv.org/abs/2211.04533"><strong>Read the official paper »</strong></a>
<br>
<a href="https://serre-lab.github.io/Harmonization/results">Explore results</a>
.
<a href="https://serre-lab.github.io/Harmonization/">Documentation</a>
.
<a href="https://serre-lab.github.io/Harmonization/models/">Models zoo</a>
.
<a href="https://serre-lab.github.io/Harmonization/evaluation/">Tutorials</a>
·
<a href="https://arxiv.org/abs/1805.08819">Click-me paper</a>
</p>


## Paper summary

<img src="docs/assets/big_picture_left.jpg" width="45%" align="right">

The many successes of deep neural networks (DNNs) over the past decade have largely been driven by computational scale rather than insights from biological intelligence. Here, we explore if these trends have also carried concomitant improvements in explaining visual strategies underlying human object recognition. We do this by comparing two related but distinct properties of visual strategies in humans and DNNs: _where_ they believe important visual features are in images and _how_ they use those features to categorize objects. Across 85 different DNNs and three independent datasets measuring human visual strategies on ImageNet, we find a trade-off between DNN top-1 categorization accuracy and their alignment with humans. _State-of-the-art_ DNNs are progressively becoming _less aligned_ with humans. We rectify this growing issue by introducing the harmonization procedure: a general-purpose training routine that aligns DNN and human visual strategies while improving object classification performance.

### Aligning the Gradients

<img src="docs/assets/qualitative_figure.png" width="100%" align="center">

Human and DNNs rely on different features to recognize objects. In contrast, our neural
harmonizer aligns DNN feature importance with humans. Gradients are smoothed from both humans
and DNNs with a Gaussian kernel to improve visualization.

### Breaking the trade-off between performance and alignment

<img src="docs/assets/imagenet_results.png" width="100%" align="center">

The trade-off between DNN performance and alignment with human feature importance from the _ClickMe_ dataset. Human feature alignment is the mean Spearman correlation between human and DNN feature importance maps, normalized by the average inter-rater alignment of humans. The grey-shaded region illustrates the convex hull of the trade-off between ImageNet accuracy and human feature alignment. All the models trained with the harmonization procedure are more accurate and aligned than versions of those models trained only for classification. Arrows denote a shift in performance after training with the harmonization procedure.

## 🗞️ Citation

If you use or build on our work as part of your workflow in a scientific publication, please consider citing the [official paper](https://arxiv.org/abs/2211.04533):

```
@article{fel2022aligning,
  title={Harmonizing the object recognition strategies of deep neural networks with humans},
  author={Fel, Thomas and Felipe, Ivan and Linsley, Drew and Serre, Thomas},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

Moreover, this paper relies heavily on previous work from the Lab, notably [Learning What and Where to Attend](https://arxiv.org/abs/1805.08819) where the ambitious ClickMe dataset was collected.

```
@article{linsley2018learning,
  title={Learning what and where to attend},
  author={Linsley, Drew and Shiebler, Dan and Eberhardt, Sven and Serre, Thomas},
  journal={International Conference on Learning Representations (ICLR)},
  year={2019}
}
```

## Tutorials

**Evaluate your own model (pytorch and tensorflow)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Mp0vxUcIsX1QY-_Byo1LU2IRVcqu7gUl) 
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/230px-Tensorflow_logo.svg.png" width=35>
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bttp-hVnV_agJGhwdRRW6yUBbf-eImRN) 
<img src="https://pytorch.org/assets/images/pytorch-logo.png" width=35>

## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.
