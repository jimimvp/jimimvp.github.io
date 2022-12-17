---
title: Frigging Diffusion Models
categories: blog
layout: math_post
---

This post is a about diffusion models that have received quite a bit attention recently and such as with DALLE [1] or IMAGEN[2] in a text to image use-case.
For me the motivation was to rather understand the underlying mechanics of the method that is surprisingly enough intuitive, but has some non-zero mathematics requirement to be "rigorous" about it.
The main paper that motivates this blog post is the (Denoising Diffusion Probabilistic Models)[https://arxiv.org/abs/2006.11239] paper from Ho et al.(2020).
As of this moment, the paper is quite recent so I dare to say that this is a very novel method to `learning generative models`.

Of course, I am not the first one to attempt to write an explanatory blog post about diffusion models! There's some pretty cool stuff out there:
* [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
* [https://yang-song.github.io/blog/2021/score/](https://yang-song.github.io/blog/2021/score/)
* [https://colab.research.google.com/drive/1SeXMpILhkJPjXUaesvzEhc3Ke6Zl_zxJ?usp=sharing#scrollTo=XCR6m0HjWGVV](https://colab.research.google.com/drive/1SeXMpILhkJPjXUaesvzEhc3Ke6Zl_zxJ?usp=sharing#scrollTo=XCR6m0HjWGVV)

But of course, this blog post is the best (ha-ha). 
What can make the above mentioned tutorials daunting is some mathematical maturity, if you are scared of stochastic differential equations and a lot of Gaussian transformations, perhaps a more intuitive explanation is for you, in which case you have come to the right place.
I'll assume that we are all "relatively" tabula rasa.

So, let's start with the simplest problem setting. 
What is the task? Generative modelling.
What do we have? Access to samples $\rx \sim p(\rx)$.
But we don't have access to $p(\rx)$ itself, and we want to generate samples from it by ourselves, for whatever weird reason.
There are all sorts of things that you can do to learn $p(x)$, some are for example training a Variational Autoencoder (VAE), a Normalizing Flow (NF) or adversarially training a Geneative Adversarial Network (GAN).
Here are some facts about those:
* With VAEs we can generate samples, but we can't evaluate likelihoods.  VAEs have long been overtaken by GANs with regards to sample quality. The benefit is that we can go from a low-dimensional latent variable $\rz$ to a higher dimensional variable $\rx$.
* With GANs we can't evaluate likelihoods and there are known to be very very difficult to train, but generate some kickass samples in terms of quality. GANs also have the property that we don't care about the dimension of the latent $\rz$.
* NFs are just awesome, by making transformations invertible, you can evaluate the likelihood of a sample, but this comes at an expense of being expensive in terms of likelihood evaluation or sampling, since NFs rely on some autoregressive prediction to make them more expressive.

Now, here come diffusion models.
The key motivation to diffusion models is as in any other wonderful thing in machine learning, the Gaussian distribution. In order to make this work, we'll deal with a latent variable $\rz$ that has the same dimensionality as $\rx$.
It turns out, if you add up a bunch of Gaussians, you get a Gaussian.
If you multiply a Gaussian by a scalar, you get another Gaussian.
Conditioning a multivariate Gaussian is a Gaussian and a marginal of a multivariate Gaussian is a Gaussian. Gaussian Gaussian Gaussian.
Brownian motion on the other hand is something that comes from adding a bunch of independent Gaussian in time (short version, long version has a definition as a stochastic process).
Why am I mentioning Brownian motion? Because the key idea is to add noise of some variance, lets call the variance $\beta$ for multiple consecutive steps to the sample $\rx$.
In the limit, by adding Gaussian noise to $\rx$, we obtain a simple distribution that we are familiar with, the Gaussian with 0 mean and unit variance $\gN(0, I)$.
Armed with this knowledge, we can write down the distribution of the "ruined" $x$ at time step $t$.

$$
q(\rx_t \vert \rx_{t-1})  = \gN(\sqrt{1 - \beta} \rx_{t-1}, \beta I).
$$


Now, we call this the `forward diffusion process`.
Obviously the forward process might not be that interesting, after all, it doesn't give us samples from $p(\rx)$ right? It kind of ruins them, phew...
Not to worry, we'll come to the sampling part later.
There are a couple of things here to make a note of.
We don't our $\rx$ to degrade fast into pure noise, so we want the variance $\beta$ to be relatively small (something smaller than 1), this also depends on the variance of $\rx$ what small actually means.
The other thing is this $\sqrt{1 - \beta}$ term multiplying the mean at time step $t$.
This term is there to keep the random walk from exploding (also called the drift coefficient).
Let's for now observer the properties of this diffusion process first.
First of all, for a fixed variance $\beta$ at each time step $t$ we can nicely compute in **closed form**, since we're just adding a bunch of Gaussians.
Let $\alpha = 1 - \beta$, then

$$
q(\rx_t \vert \rx_0)  = \gN(\sqrt{\alpha^t} \rx_{t-1}, (1-\alpha)^t I).
$$

We have a "closed-form" distribution for $\rx_t$ given the original sample $\rx$.
Note, nothing stops you to have $\beta$ be time-dependent (indeed it turns out it is useful to make it as such and leads to better sample quality), or learned in some way, I decided to hold it fixed for simplicity of presentation. 

## What About the Reverse Direction?
<br>
As mentioned earlier, if we keep adding Gaussian noise to the original sample with the drift term, we'll converge to $\gN(0,\, I)$, this is a key insight, since in theory if we could learn the reverse transformations we are able to "reconstruct" $\rx$ or more precisely, sample from $p(\rx)$.
There is an interesting connection to other generative models here, first of all, it's again about this latent $\rz$. Note that, I use $\rx_T$ and $\rz$ interchangeably, it just so happens that $\rx_T$ is going to be distributed something close to how $\rz$ is distributed, in the end we sample $\rx_T$ for the reverse direction the same as we would sample $\rz$.
Second, the close connection to VAEs is indeed the encoding step, the difference is that we don't learn the encoding step, we rather just use normally distributed noise to project to a zero mean and unit variance normal distribution. 
To do anything useful, we need to learn the distributions $q(\rx_{t-1} \vert \rx_t)$.
As it turns out, if the noise variances on the forward direction are small enough, then $q(\rx_{t-1} \vert \rx_t)$ is also Gaussian.

The key question now is, how would a reverse procedure look like if we have access to $q(\rx_t \vert \rx_{t-1})$. 
We would do something like this:

1. Sample $\rx_T$ from $\gN(0,\, I)$, initialize $t=T$
2. While $t\geq0$ do $\rx_{t-1} \sim q(\rx_{t-1} \vert \rx_t)$; $t$-=1
3. Return $\rx_0$

Indeed, there's a closed-form expression in case we can condition on $\rx_0$. $q(\rx_{t-1} \vert \rx_{t}, \rx_{0})$, it's a Gaussian with mean

$$
\mu_t = \frac{1}{\sqrt{\alpha}} (x_t - \frac{\beta}{\sqrt{1- \alpha^t}}\rz_t).
$$

The variance resolves to

$$
\beta_t = \frac{\sqrt{\alpha}(1-\alpha^{t-1})}{1-\alpha^t}\rx_t - \frac{\sqrt{\alpha^{t-1}}\beta}{(1-\alpha)\sqrt{\alpha^t}}(\rx_t - \sqrt{1-\alpha}\rz_t).
$$

So the problem here is that when we generate the samples, we don't have access to $\rx_0$, duh! That's the thing that we want to get in the first place.
But at least it's now clear, we want to approximate  $q(\rx_{t-1} \vert \rx_{t}, \rx_{0})$ with some distribution $p_\theta(\rx_{t-1} \vert \rx_{t})$.
Now we can look at our ML toolbox and pull out variational inference, derive a variational lower bound and boom, we're in business, starting with the reverse-KL divergence

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
&= \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ].
\end{aligned}
$$.

That would be the "fancy" theory that derives the bound.
What do we need now? We need a model for $p_\theta(\rx_t \vert \rx_{t+1})$.
Note that there's is a time dependence here, so we need to also provide time information (which diffusion step we're predicting for), to our model.
Without further ado, we'll model our reverse diffusion step as Gaussian with

$$
\begin{aligned}
&\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \mathbf{z}_\theta(\mathbf{x}_t, t) \Big) \\
&\text{and simply predicting covariance }\, \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)
\end{aligned}
$$


Simplified version (very intuitive):

$$
L_t^\text{simple} = \mathbb{E}_{\mathbf{x}_0, \mathbf{z}_t} \Big[\|\mathbf{z}_t - \mathbf{z}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\mathbf{z}_t, t)\|^2 \Big]
$$