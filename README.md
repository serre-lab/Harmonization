
# Aligning Machine & Human Vision

_Thomas Fel*, Ivan Felipe Rodriguez*, Drew Linsley*, Thomas Serre_

<p align="center">
<a href="todo"><strong>Read the official paper »</strong></a>
<br>
<a href="https://serre-lab.github.io/harmonization/results">Explore results</a>
.
<a href="https://serre-lab.github.io/Harmonization/">Documentation</a>
.
<a href="https://serre-lab.github.io/harmonization/models">Models zoo</a>
.
<a href="todo">Tutorials</a>
</p>

## Paper summary

<img src="docs/assets/big_picture_left.jpg" width="45%" align="right">

The many successes of deep neural networks (DNNs) over the past decade have largely been driven by computational scale rather than insights from biological intelligence. Here, we explore if these trends have also carried concomitant improvements in explaining visual strategies underlying human object recognition. We do this by comparing two related but distinct properties of visual strategies in humans and DNNs: _where_ they believe important visual features are in images and _how_ they use those features to categorize objects. Across 85 different DNNs and three independent datasets measuring human visual strategies on ImageNet, we find a trade-off between DNN top-1 categorization accuracy and their alignment with humans. _State-of-the-art_ DNNs are progressively becoming _less aligned_ with humans. We rectify this growing issue by introducing the harmonization procedure: a general-purpose training routine that aligns DNN and human visual strategies while improving object classification performance.

### Aligning the Gradients

<img src="docs/assets/qualitative_figure.jpg" width="100%" align="center">

Human and DNNs rely on different features to recognize objects. In contrast, our neural
harmonizer aligns DNN feature importance with humans. Gradients are smoothed from both humans
and DNNs with a Gaussian kernel to improve visualization.

### Breaking the trade-off between performance and alignment

<img src="docs/assets/imagenet_results.png" width="100%" align="center">

The trade-off between DNN performance and alignment with human feature importance from the _ClickMe_ dataset. Human feature alignment is the mean Spearman correlation between human and DNN feature importance maps, normalized by the average inter-rater alignment of humans. The grey-shaded region illustrates the convex hull of the trade-off between ImageNet accuracy and human feature alignment. All the models trained with the harmonization procedure are more accurate and aligned than versions of those models trained only for classification. Arrows denote a shift in performance after training with the harmonization procedure.

## 🗞️ Citation

If you use or build on our work as part of your workflow in a scientific publication, please consider citing the [official paper](todo):

```
@article{fel2022aligning,
  title={Aligning deep neural network strategies for object recognition with humans},
  author={Fel, Thomas and Felipe, Ivan and Linsley, Drew and Serre, Thomas},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```
## 📝 License

The package is released under <a href="https://choosealicense.com/licenses/mit"> MIT license</a>.