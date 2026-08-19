---
title: "Network Pruning for Weakly Supervised Object Localization"
draft: false
description: "Investigating how neural network pruning affects object localization and class activation maps in convolutional neural networks."
tags:
  - Computer Vision
  - Deep Learning
  - PyTorch
  - Explainable AI
  - Model Pruning
  - WSOL
---

# Network Pruning for Weakly Supervised Object Localization

Can making a neural network smaller actually make it **better at understanding where an object is?**

In this research project, I investigated how **network pruning** affects Weakly Supervised Object Localization (WSOL), with a particular focus on Class Activation Mapping methods.

The experiments revealed a counterintuitive result:

> **Pruning can improve object localization even when classification accuracy decreases.**

This effect was particularly pronounced for the heavily overparameterized VGG16 architecture.

[Download the full research paper](/files/Liam-Leirs-WSOL-paper.pdf)

---

## The Problem

Weakly Supervised Object Localization aims to identify the location of an object using models trained only with **image-level class labels**.

Unlike conventional object detection, bounding-box annotations are not required during training.

Instead, techniques such as Class Activation Mapping can reveal which parts of an image contribute most strongly to a network's prediction.

There is a problem, however.

A classification network does not need to understand the entire object to classify it correctly. It may learn to focus on only a small, highly discriminative region.

For example, a network identifying a dog breed might learn that the dog's face or ears contain enough information to make the correct prediction.

That can be excellent for classification while being poor for localization.

This project investigated whether **removing parameters through network pruning could change this behavior**.

---

## Research Questions

The project focused on three questions:

1. How does network pruning affect localization performance in CAM-based WSOL?
2. What trade-offs exist between pruning level, classification accuracy, and localization performance?
3. How does pruning affect the spatial quality and interpretability of activation maps?

---

## Experimental Setup

I designed an experimental pipeline combining different neural network architectures, datasets, pruning strategies, and localization techniques.

### Architectures

Two convolutional neural networks were evaluated:

- **ResNet18**
- **VGG16**

The architectures differ substantially in capacity. ResNet18 contains approximately 11.7 million parameters, while VGG16 contains roughly 138 million.

This made it possible to investigate whether larger, more redundant networks respond differently to pruning.

### Datasets

Experiments were conducted on two fine-grained image classification datasets:

- **CUB-200-2011** — 200 bird species
- **Stanford Dogs** — 120 dog breeds

Both provide bounding-box annotations, allowing localization performance to be evaluated even though those annotations are not used to train the classifier.

---

## Pruning Strategies

Three pruning approaches were investigated.

### Magnitude Pruning

Magnitude pruning removes individual weights with small absolute values, producing a sparse network while maintaining the original architecture.

I evaluated sparsity levels of:

`50%` · `70%` · `90%`

### Filter-Norm Pruning

Filter-Norm pruning removes complete convolutional filters based on their norm.

I evaluated:

`10%` · `30%` · `50%`

### Filter Pruning via Geometric Median

FPGM attempts to identify filters that contain redundant information and removes them from the network.

Again, I evaluated:

`10%` · `30%` · `50%`

After pruning, the networks were fine-tuned before evaluation.

---

## Measuring Where the Network Looks

For every trained model, I generated localization maps using three CAM-based methods:

**Grad-CAM** uses gradients to identify convolutional features that contribute strongly to a prediction.

**Layer-CAM** retains more spatial information by using spatially resolved gradients.

**Score-CAM** estimates activation importance through changes in the model's prediction score rather than directly using gradients.

The resulting activation maps were converted into predicted bounding boxes and compared with the ground-truth object locations.

---

## Visualization 1 — What Pruning Changes

![Comparison of activation maps before and after network pruning](/projects/wsol/cam-comparison.png)

_Example Class Activation Maps before and after pruning. The pruned network produces a more spatially complete representation of the object rather than concentrating exclusively on a small discriminative region._

This qualitative difference was one of the most interesting outcomes of the project.

Before pruning, activation maps could become highly concentrated around small regions that were useful for classification.

After pruning, activation frequently became more spatially distributed across the object.

---

## Evaluation

Localization was measured using three WSOL metrics:

- **GT-Known Localization Accuracy** — evaluates localization when the correct class is known.
- **Top-1 Localization Accuracy** — requires both correct classification and correct localization.
- **MaxBoxAccV2** — evaluates bounding-box localization across multiple activation thresholds.

Using several metrics was important because classification and localization performance do not necessarily move in the same direction.

---

# Results

## Classification Accuracy Decreases...

As expected, aggressive pruning generally reduced classification accuracy.

The effect differed substantially between architectures.

**ResNet18 was relatively sensitive to pruning**, particularly when complete filters were removed.

VGG16 was much more resistant.

Its much larger parameter count appears to provide considerable redundancy, allowing substantial parts of the network to be removed while retaining useful representations.

But classification accuracy only tells part of the story.

---

## ...While Localization Can Improve

One of the central findings of the project was that:

> **Classification performance and localization performance are not strongly coupled.**

A network can become worse at predicting the correct class while simultaneously becoming **better at identifying the complete object**.

This effect was especially visible with VGG16.

---

## Visualization 2 — VGG16 on Stanford Dogs

![Localization performance of VGG16 on Stanford Dogs across pruning levels](/projects/wsol/vgg16-stanford-dogs.png)

_Localization performance of VGG16 on Stanford Dogs. Despite removing substantial network capacity, several pruned configurations improve localization performance._

The Stanford Dogs experiments produced some of the clearest evidence of the effect.

VGG16 could tolerate substantial pruning, and localization performance recovered or improved as redundant network capacity was removed.

The improvement also appeared across different CAM techniques rather than being restricted to a single localization method.

---

## ResNet18 vs. VGG16

The two architectures responded differently.

### ResNet18

ResNet18 maintained relatively strong localization performance under moderate pruning, but aggressive structured pruning eventually degraded its performance.

Because ResNet18 is already relatively compact, there appears to be less redundant capacity available for removal.

### VGG16

VGG16 behaved very differently.

During classification-oriented fine-tuning, its localization performance could deteriorate as the network learned increasingly specialized discriminative features.

Pruning could partially reverse this effect.

Removing redundant capacity encouraged the model to produce more spatially complete object representations.

---

# Why Can Pruning Improve Localization?

One interpretation of the results is that pruning acts as a form of **implicit regularization**.

Large neural networks have enough capacity to learn extremely specialized features.

For fine-grained classification, these might correspond to very small but highly discriminative regions.

A dog classifier, for example, may discover that the shape of an ear or part of the face is enough to distinguish one breed from another.

For classification, that's perfectly useful.

For localization, it isn't.

Removing redundant weights or filters limits some of this specialized capacity.

The model may therefore need to rely on features that are more broadly distributed across the object.

The resulting activation maps can become:

- more object-centered;
- more spatially coherent;
- less concentrated on isolated discriminative regions;
- better aligned with the object's full spatial extent.

This leads to an important distinction:

> **The best classifier is not necessarily the best localizer.**

---

## Pruning Is More Than Model Compression

Network pruning is traditionally associated with reducing:

- parameter count;
- memory requirements;
- computational requirements.

The results of this project suggest another perspective.

Pruning can also alter the **representations learned by a neural network**.

In some cases, these changes can improve localization and interpretability even when classification performance decreases.

This makes pruning interesting not only as an optimization technique, but potentially as a tool for studying **representation learning and Explainable AI**.

---

## Key Findings

- Moderate pruning can preserve strong localization performance despite decreasing classification accuracy.
- Classification accuracy and localization quality are only weakly correlated.
- ResNet18 is more sensitive to aggressive pruning than VGG16.
- VGG16 contains substantial redundant capacity.
- Pruning can improve localization in overparameterized models.
- The strongest effects were observed with VGG16 on Stanford Dogs.
- Pruned models can produce more spatially coherent and object-centered activation maps.
- The effect appears across multiple CAM methods.
- Pruning may act as an implicit regularizer against excessive reliance on highly discriminative local features.

---

## What I Learned

Beyond the individual results, this project gave me experience designing a relatively large computer vision experiment with several interacting dimensions:

- multiple neural network architectures;
- multiple datasets;
- several pruning strategies;
- different pruning levels;
- multiple CAM algorithms;
- classification and localization metrics.

One of the biggest lessons was the importance of **looking beyond a single evaluation metric**.

If these models had only been compared using classification accuracy, some of the most interesting effects of pruning would have remained invisible.

Visualizing what the networks actually focused on revealed a very different story.

---

## Technologies

`Python` · `PyTorch` · `Computer Vision` · `CNNs` · `ResNet18` · `VGG16` · `Grad-CAM` · `Layer-CAM` · `Score-CAM` · `Network Pruning` · `WSOL`

---

## Research Paper

For the complete methodology, quantitative results, hyperparameters, and additional experiments:

### On the Effect of Network Pruning on Weakly Supervised Object Localization

**Liam Leirs**  
University of Antwerp

[Download the full paper (PDF)](/files/Liam-Leirs-WSOL-paper.pdf)
