---
title: "Revisiting GRU4Rec: A Controlled Comparison of Recommender System Implementations"
draft: false
description: "Investigating why different implementations of GRU4Rec produce substantially different recommendation performance under seemingly comparable experimental settings."
tags:
  - Recommender Systems
  - Deep Learning
  - PyTorch
  - GRU4Rec
  - Machine Learning
  - Reproducibility
---

# Revisiting GRU4Rec Implementations

What happens when two implementations of supposedly the same machine learning algorithm produce very different results?

My Master's thesis investigates this problem using **GRU4Rec**, a recurrent neural network architecture for session-based recommendation.

Previous experimental comparisons reported substantial performance differences between the **official GRU4Rec implementation** and the implementation available in **RecPack**.

Rather than treating this simply as evidence that one implementation was better, I investigated a different question:

> **Where does the performance difference actually come from?**

Through code-level analysis, controlled experiments, implementation modifications, and independent hyperparameter optimization, I investigated how seemingly small implementation and evaluation choices can substantially affect recommender-system benchmarks.

[Download the full Master's thesis (PDF)](/files/Liam-Leirs-Master-Thesis-GRU4Rec.pdf)

---

## The Problem

Reproducibility is a major challenge in machine learning.

Research papers often compare algorithms using benchmark datasets and report which model performs best.

But an algorithm is not just an equation.

Its actual performance also depends on:

- implementation details;
- data preprocessing;
- loss functions;
- negative sampling;
- optimization;
- hyperparameters;
- evaluation protocols.

As a result, two implementations carrying the same algorithm name can behave very differently.

GRU4Rec provided an interesting case study for investigating this problem.

---

## GRU4Rec

GRU4Rec is a neural recommender system designed for **session-based recommendation**.

Instead of relying on long-term user profiles, it models sequences of interactions within individual sessions.

A simplified session might look like:

```text
Running Shoes → Sports Socks → Running Watch → ?
```

The model observes the sequence and attempts to predict which item the user is most likely to interact with next.

GRU4Rec uses a **Gated Recurrent Unit (GRU)** to maintain a hidden representation of the session as interactions occur.

```text
Item₁ ──► GRU ──► Item₂ ──► GRU ──► Item₃
             │                 │
             ▼                 ▼
        Hidden state      Hidden state
                               │
                               ▼
                     Next-item prediction
```

This makes GRU4Rec particularly suitable for situations where recommendations depend strongly on a user's recent sequence of actions.

---

## The Reproducibility Question

An earlier comparison found a substantial difference between:

- the official GRU4Rec implementation;
- the implementation provided by RecPack.

At first glance, this could suggest that the RecPack implementation simply performs worse.

But comparing machine learning implementations is rarely that straightforward.

The implementations differed in several ways beyond the underlying GRU architecture.

My thesis therefore investigated whether the reported performance gap reflected a fundamental limitation of RecPack's GRU4Rec implementation or whether it could instead be explained by differences in implementation and experimental setup.

---

## Research Approach

I approached the problem in several stages.

### 1. Code-Level Comparison

I first examined both GRU4Rec implementations to identify differences in:

- model architecture;
- training procedure;
- loss computation;
- negative sampling;
- initialization;
- optimization;
- data handling.

This provided a map of potential causes for the observed performance difference.

---

### 2. Controlled Experimental Setup

The implementations were then evaluated under more closely aligned experimental conditions.

The goal was to remove as many confounding factors as possible.

Rather than asking:

> "Which library produces the highest number?"

the more useful question became:

> "What happens when both implementations are given comparable opportunities to perform well?"

---

### 3. Modifying RecPack

Where meaningful implementation differences were identified, I modified the RecPack version of GRU4Rec to more closely reproduce relevant behavior from the official implementation.

This allowed individual implementation choices to be investigated experimentally rather than merely discussed theoretically.

---

### 4. Independent Hyperparameter Optimization

A particularly important part of the methodology was **independent hyperparameter optimization**.

Different implementations do not necessarily perform optimally under the same hyperparameters.

Using parameters optimized for one implementation can therefore unintentionally favor it.

Both implementations were given their own optimization process so that the comparison reflected their achievable performance rather than the suitability of one shared configuration.

---

# Experiments

The implementations were compared across multiple recommendation datasets under controlled experimental settings.

Performance was evaluated using ranking-based recommendation metrics.

A central metric was **Recall@20**, which measures how often the relevant next item appears among the model's top 20 recommendations.

The experiments were designed not only to identify the strongest configuration, but to understand **why performance changed** as individual experimental choices were modified.

---

## Closing the Performance Gap

The controlled experiments showed that the previously observed gap was **not simply an inherent difference between the two implementations**.

After better aligning the experimental setup and independently optimizing both systems, RecPack became considerably more competitive.

In particular, configurations using **Cross-Entropy loss** performed strongly across several datasets.

However, the implementations did not become completely equivalent.

Meaningful performance differences remained, particularly on larger datasets such as Taobao.

---

## Performance Is Only Part of the Story

Recommendation quality was not the only difference between the implementations.

Computational efficiency also mattered.

Some configurations could achieve competitive recommendation performance while differing substantially in training cost.

This became particularly relevant on larger datasets.

---

# What Caused the Differences?

The experiments showed that there was no single switch responsible for the entire performance gap.

Instead, performance emerged from the interaction of multiple design choices.

These included differences in the training pipeline, loss formulation, optimization, sampling behavior, and hyperparameter configuration.

This makes the comparison more interesting than simply determining a "winner."

The results demonstrate that:

> **Benchmark performance belongs to an entire experimental pipeline, not just to the algorithm name attached to it.**

---

## Why Hyperparameter Optimization Matters

One particularly important lesson was the effect of hyperparameter optimization.

Suppose implementation A and implementation B both implement GRU4Rec.

Using exactly the same hyperparameters might initially seem like the fairest comparison.

But it may actually create an unfair experiment.

If the implementations differ internally, the optimal learning rate, regularization, sampling configuration, or other parameters may also differ.

Independent optimization therefore provides a better estimate of what each implementation is actually capable of.

This was an important factor in reducing the previously observed performance difference.

---

# Key Findings

The main findings of the thesis were:

- Different implementations of the same recommender algorithm can produce substantially different benchmark results.
- These differences cannot automatically be attributed to the underlying algorithm.
- Implementation and experimental details can have a large effect on measured performance.
- Hyperparameters should be optimized independently when implementations differ meaningfully.
- Better experimental alignment substantially reduced the previously reported gap between RecPack and the official GRU4Rec implementation.
- RecPack became competitive on several datasets, particularly when using Cross-Entropy loss.
- Meaningful differences remained on larger datasets such as Taobao.
- Recommendation quality and computational efficiency should both be considered when comparing implementations.
- Reproducible benchmarking requires reporting more than the final evaluation score.

---

## A Broader Reproducibility Lesson

Although this thesis focuses specifically on GRU4Rec, the underlying problem applies much more broadly to machine learning.

It is tempting to read a benchmark table as:

```text
Model A > Model B
```

But the actual comparison is closer to:

```text
Implementation
+ preprocessing
+ optimization
+ hyperparameters
+ evaluation protocol
──────────────────────────
          result
```

Changing any of these components can change the conclusion.

This makes careful experimental design particularly important when reproducing or comparing machine learning research.

---

## What I Learned

This thesis gave me experience with a different side of machine learning than simply building and training models.

A large part of the work involved understanding **why experimental results occurred**.

That required:

- reading and comparing existing implementations;
- understanding unfamiliar research code;
- designing controlled experiments;
- modifying an existing ML framework;
- building reproducible preprocessing pipelines;
- performing hyperparameter optimization;
- running large experimental comparisons;
- analyzing recommendation metrics;
- investigating unexpected results;
- separating algorithmic effects from implementation effects.

Perhaps the most important lesson was that reproducing machine learning research requires considerably more than rerunning published code.

Small implementation and experimental decisions can have surprisingly large consequences.

---

## Technologies

`Python` · `PyTorch` · `RecPack` · `GRU4Rec` · `Recommender Systems` · `Deep Learning` · `Hyperparameter Optimization` · `Session-Based Recommendation` · `Experimental ML`

---

## Master's Thesis

For the complete methodology, implementation analysis, experimental results, and discussion:

### Revisiting the Comparison of GRU4Rec Implementations

**Liam Leirs**  
Master of Data Science and Artificial Intelligence  
University of Antwerp, 2026

[Download the full Master's thesis (PDF)](/files/Liam-Leirs-Master-Thesis-GRU4Rec.pdf)

<!-- Add this if/when the thesis code is public:
[View source code on GitHub](YOUR_GITHUB_URL)
-->
