---
title: Frigging Diffusion Models
categories: blog
layout: math_post
---

This post is about diffusion models, which have received quite a bit of attention with systems like DALL-E and Imagen in text-to-image generation.
For me, the motivation was to understand the underlying mechanics of the method — which is surprisingly intuitive, yet requires a bit of care to write down rigorously.
The main paper that motivates this blog post is [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) by Ho et al. (2020).

Of course, I am not the first one to attempt to write an explanatory blog post about diffusion models! There's some pretty cool stuff out there:
* [Lilian Weng — What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
* [Yang Song — Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.github.io/blog/2021/score/)
* [Annotated Diffusion Colab](https://colab.research.google.com/drive/1SeXMpILhkJPjXUaesvzEhc3Ke6Zl_zxJ?usp=sharing#scrollTo=XCR6m0HjWGVV)

What can make some of the tutorials above daunting is the mathematical machinery — if you're intimidated by stochastic differential equations and pages of Gaussian identities, perhaps a more intuitive walkthrough is for you. I'll assume we're starting relatively *tabula rasa*.

## Problem Setup

What is the task? Generative modeling.
What do we have? Access to samples $$\rx \sim p(\rx)$$.
We don't have access to $$p(\rx)$$ itself, and we want to generate new samples from it.
There are all sorts of ways to learn $$p(\rx)$$, such as training a Variational Autoencoder (VAE), a Normalizing Flow (NF), or adversarially training a Generative Adversarial Network (GAN):
* **VAEs** let us generate samples from a low-dimensional latent $$\rz$$, though we can only optimize a lower bound on (not evaluate) exact likelihoods.
* **GANs** avoid likelihood evaluation altogether and can produce very sharp samples, though adversarial training is notoriously tricky to stabilize.
* **Normalizing Flows** make every transformation invertible so we can evaluate exact likelihoods, at the cost of architectural constraints that can be expensive to scale.

Now, enter diffusion models.
The key ingredient of diffusion models is, as in so many wonderful things in machine learning, the Gaussian distribution. Here we'll work with a latent variable $$\rz$$ that has the same dimensionality as $$\rx$$.
Recall a few basic properties of Gaussians:
* Adding independent Gaussians yields a Gaussian.
* Scaling a Gaussian by a constant yields a Gaussian.
* Both conditionals and marginals of a multivariate Gaussian are Gaussian.

Brownian motion, similarly, arises from adding independent Gaussian increments over time.
Why mention Brownian motion? Because the core idea is to add Gaussian noise of some variance $$\beta$$ over $$T$$ consecutive steps to a data sample $$\rx_0$$, gradually erasing all structure until we arrive at a simple distribution we know how to sample from: the standard normal $$\gN(0, I)$$.

## The Forward Diffusion Process

We define the transition distribution of the "noised" sample at time step $$t$$ as:

$$
q(\rx_t \vert \rx_{t-1}) = \gN(\sqrt{1 - \beta}\,\rx_{t-1}, \beta I).
$$

We call this the **forward diffusion process**.
Obviously, the forward process by itself doesn't give us new samples from $$p(\rx)$$ — it destroys them!
Two details to note right away:
* We keep $$\beta \in (0, 1)$$ small so $$\rx_0$$ doesn't collapse into pure noise in a single step.
* The $$\sqrt{1 - \beta}$$ contraction factor keeps the variance from growing unboundedly as $$t$$ increases.

Because each step is a linear Gaussian transformation, we can write $$\rx_t = \sqrt{\alpha}\,\rx_{t-1} + \sqrt{1 - \alpha}\,\mathbf{z}_{t-1}$$ with $$\alpha = 1 - \beta$$ and $$\mathbf{z}_{t-1} \sim \gN(0, I)$$. Unrolling this one step further:

$$
\begin{aligned}
\rx_t &= \sqrt{\alpha}\big(\sqrt{\alpha}\,\rx_{t-2} + \sqrt{1 - \alpha}\,\mathbf{z}_{t-2}\big) + \sqrt{1 - \alpha}\,\mathbf{z}_{t-1} \\
&= \alpha\,\rx_{t-2} + \sqrt{\alpha(1 - \alpha)}\,\mathbf{z}_{t-2} + \sqrt{1 - \alpha}\,\mathbf{z}_{t-1}.
\end{aligned}
$$

Since the sum of two independent zero-mean Gaussians with variances $$\sigma_1^2$$ and $$\sigma_2^2$$ is a Gaussian with variance $$\sigma_1^2 + \sigma_2^2$$, the two noise terms merge into a single Gaussian with variance $$\alpha(1 - \alpha) + (1 - \alpha) = 1 - \alpha^2$$. Repeating this induction down to $$t=0$$ gives the **closed-form** marginal for any step $$t$$:

$$
q(\rx_t \vert \rx_0) = \gN(\sqrt{\alpha^t}\,\rx_0, (1 - \alpha^t) I).
$$

Equivalently, $$\rx_t = \sqrt{\alpha^t}\,\rx_0 + \sqrt{1 - \alpha^t}\,\mathbf{z}$$ for $$\mathbf{z} \sim \gN(0, I)$$.
Note that nothing stops you from making $$\beta_t$$ time-dependent — indeed, a noise schedule $$\beta_1, \dots, \beta_T$$ improves sample quality, in which case $$\alpha^t$$ is simply replaced by $$\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$$. I'm holding $$\beta$$ fixed here for simplicity.

## What About the Reverse Direction?

As $$T$$ grows large, $$\alpha^T \to 0$$, so $$q(\rx_T \vert \rx_0) \to \gN(0, I)$$. This is the key insight: if we can learn the reverse transitions $$q(\rx_{t-1} \vert \rx_t)$$, we can start from pure noise and iteratively denoise it back into a sample from $$p(\rx)$$.
Notice the close connection to VAEs: you can view $$\rx_T$$ as the latent variable $$\rz$$, except the encoding process $$q(\rx_t \vert \rx_{t-1})$$ is fixed rather than learned.
When the forward step variances $$\beta$$ are small enough, the true reverse transition $$q(\rx_{t-1} \vert \rx_t)$$ is also approximately Gaussian.

If we had access to $$q(\rx_{t-1} \vert \rx_t)$$, sampling would look like this:

1. Sample $$\rx_T \sim \gN(0, I)$$, initialize $$t = T$$.
2. While $$t \ge 1$$, sample $$\rx_{t-1} \sim q(\rx_{t-1} \vert \rx_t)$$ and set $$t \leftarrow t - 1$$.
3. Return $$\rx_0$$.

While $$q(\rx_{t-1} \vert \rx_t)$$ is intractable in general (it requires integrating over the whole data distribution), it *does* have a closed-form Gaussian expression if we also condition on $$\rx_0$$. By Bayes' rule and the Markov property:

$$
q(\rx_{t-1} \vert \rx_t, \rx_0) = \frac{q(\rx_t \vert \rx_{t-1})\, q(\rx_{t-1} \vert \rx_0)}{q(\rx_t \vert \rx_0)}.
$$

Since all three terms on the right are known Gaussians, completing the square in the exponent with respect to $$\rx_{t-1}$$ gives $$q(\rx_{t-1} \vert \rx_t, \rx_0) = \gN(\tilde{\mu}_t(\rx_t, \rx_0), \tilde{\beta}_t I)$$ with mean

$$
\tilde{\mu}_t(\rx_t, \rx_0) = \frac{\sqrt{\alpha^{t-1}}\beta}{1 - \alpha^t}\rx_0 + \frac{\sqrt{\alpha}(1 - \alpha^{t-1})}{1 - \alpha^t}\rx_t
$$

and variance

$$
\tilde{\beta}_t = \frac{1 - \alpha^{t-1}}{1 - \alpha^t}\beta.
$$

The catch is that when generating new samples at test time, we don't have $$\rx_0$$ — that's the very thing we're trying to generate!
So instead, we approximate the reverse transition with a learned parametric model $$p_\theta(\rx_{t-1} \vert \rx_t)$$.
Pulling variational inference out of our ML toolbox, we can write down the variational lower bound (VLB) starting from the KL divergence over trajectories:

$$
\begin{aligned}
L_\text{VLB} 
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= \mathbb{E}_q \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]\\
&= \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] \\
&= \mathbb{E}_q \Big[\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} \Big].
\end{aligned}
$$

## Parametrization, Simplified Objective, and Score Matching

Notice that in $$L_{t-1}$$ we are matching $$p_\theta(\rx_{t-1} \vert \rx_t)$$ to $$q(\rx_{t-1} \vert \rx_t, \rx_0)$$ — a KL divergence between two Gaussians!
Since $$\rx_t = \sqrt{\bar{\alpha}_t}\rx_0 + \sqrt{1 - \bar{\alpha}_t}\mathbf{z}_t$$ for $$\mathbf{z}_t \sim \gN(0, I)$$, we can solve for $$\rx_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\rx_t - \sqrt{1 - \bar{\alpha}_t}\mathbf{z}_t)$$ and substitute it into $$\tilde{\mu}_t(\rx_t, \rx_0)$$:

$$
\tilde{\mu}_t(\rx_t, \rx_0) = \frac{1}{\sqrt{\alpha_t}} \Big( \rx_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \mathbf{z}_t \Big).
$$

This motivates modeling the reverse step $$p_\theta(\rx_{t-1} \vert \rx_t)$$ as a Gaussian whose mean predicts and subtracts the noise $$\mathbf{z}_\theta(\rx_t, t)$$ (given the noisy state $$\rx_t$$ and diffusion step $$t$$):

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \mathbf{z}_\theta(\mathbf{x}_t, t) \Big), \qquad \text{with covariance } \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t).
$$

Dropping the step-dependent weighting factors in $$L_{t-1}$$ yields the simplified training objective from Ho et al. (2020), which has a remarkably clean interpretation — just predict the noise $$\mathbf{z}_t$$ that was added to $$\rx_0$$:

$$
L_t^\text{simple} = \mathbb{E}_{\mathbf{x}_0, \mathbf{z}_t} \Big[\|\mathbf{z}_t - \mathbf{z}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\mathbf{z}_t, t)\|^2 \Big].
$$

Finally, this also reveals the deep link to **score-based generative modeling**: since $$\nabla_{\rx_t} \log q(\rx_t \vert \rx_0) = -\frac{\rx_t - \sqrt{\bar{\alpha}_t}\rx_0}{1 - \bar{\alpha}_t} = -\frac{\mathbf{z}_t}{\sqrt{1 - \bar{\alpha}_t}}$$, training $$\mathbf{z}_\theta(\rx_t, t)$$ to predict the noise is equivalent (up to a known scale factor) to learning the **score function** $$\nabla_{\rx_t} \log q(\rx_t)$$ of the noisy data distribution. At sampling time, drawing $$\rx_T \sim \gN(0, I)$$ and running the reverse chain $$t = T, \dots, 1$$ acts like annealed Langevin dynamics, iteratively nudging pure noise back onto the data manifold.
