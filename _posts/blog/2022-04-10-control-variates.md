---
title: Variance Reduction via Control Variates
categories: blog
layout: math_post
---
## Control Variates

This is one of the most typical approaches to reducing variance of MC estimates. Consider the setup where we want to estimate $\E[f(x)]$ with MC, imagine that we have access to a function $h(x)$, now we can compute the expectation of the difference $\E[f(x) - h(x)]$. This might not be that interesting, since by linearity of expectation we would arrive to just $\E[f(x)] - \E[h(x)]$, but things get interesting when we estimate the quantity via MC

$$
\hat{\mu}_d = \frac{1}{n}\sum_{i=0}^n (f(x_i) - h(x_i))
$$

Further, I will use the shorthand $d = f(x) - h(x)$ to denote the difference and reduce the clutter, with $\mu_f$ and $\mu_h$ being the expected values of each function respectively. First question to ask is, is this estimator unbiased? The answer is yes, you can check this relatively easily by writing out the integral, i.e. $\E[\hat{\mu}_d]$ with replacing $x_i$ by samples, we would arrive to $\E[f(x)] - \E[h(x)]$. Next question is, and this is the far more interesting one, how does the variance of this estimator behave? So we want to look at

$$
\Var[\hat{\mu}_d] = \hat{\mu}_d^2 - \mu_d^2
$$

This is the result of the standard identity for variance. As it turns out, switching out the difference with $d$ and looking at $\mu_d$ brings us to the same results regarding error estimates and variance of the estimator as in the standard Monte Carlo case, meaning that

$$
\Var[\hat{\mu}_d] = \frac{1}{S}\Var[f(x)-h(x)].
$$

Naturally, this is a bit weird, since we are estimating the difference expectation  $\E[f(x)-h(x)]$ whereas we would like to be estimating $f(x)$. To do a sort of "prediction" based on the difference expectation, we need only to add $\E[h(x)]$. Why is this so? Because

$$
\E[f(x)-h(x)] = \E[f(x)] - \E[h(x)]
$$

by linearity of expectation. It is relatively obvious that we can kind of go both ways in estimating this expectation, we can do a Monte Carlo estimate of the difference directly, or we can do a Monte Carlo estimate of each expectation individually, though we couldn't reap the benefits of variance reduction in this case. Going back to the point, we can make a prediction of $\mu_f = \E[f(x)]$ as 

$$
\hat{\mu}_f = \hat{\mu}_d + \mu_h  
$$

Note that, even for the case when we don't have a nice closed-form expression for $\mu_h$, it still might be beneficial to estimate $\mu_h$, since the variance of that estimator could be smaller than the variance of an estimator that estimates $\mu_f$ directly. So, one might as, what is a good choice of $h(x)$? Obviously, the one that reduces the variance the most. Let's inspect the variance a bit again

<div class="d-flex justify-content-center">
    <img src="http://i.imgur.com/nPSPGxl.png"  class="w-50">
</div>

What this tells us is that:
1. The variance of this estimator with a control variate grows with the variance of $f(x)$ and $h(x)$.
2. To reduce the variance, it's beneficial to use $h(x)$ that is correlated with $f(x)$

The stronger the correlation between $f(x)$ and $h(x)$, the larger the covariance. Suppose that $h(x)$ is parametrized with a set of parameters $\theta$, then we can indeed find an optimal $h$ that is going to reduce our covariance the most. This is achievable by simple extrema finding with taking the gradient of variance with respect to $\theta$ and setting it to $0$ for example (keeping in mind convexity with respect to $\theta$), or using another standard minimization procedure.

To make all of this talking more concrete, suppose you want to estimate the mean of a Gaussian (granted, maybe a more difficult example would be more informative, but this suffices for our purposes). The true mean is $\mu=-10$, but we have access to samples $x$ from a $0$ mean Gaussian. In the code below, we have 3 options that we are using for estimating the mean based on samples:
1. Pure Monte Carlo
2. Monte Carlo with control variate $h(x) = 10x - 2$
3. Monte Carlo with control variate $h(x) = x - 2$


```python
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['lines.linewidth'] = 4

# true mean of f
muf = -10
f = lambda x: (x - 10)

muh = -2
h = lambda x: (x*10 + muh)
mu_halt = -5
h_alt = lambda x: (x - 5)


err1 = []
err2 = []
err3 = []

samples = np.logspace(1, 15, 30, base=2).astype(np.int32)
for s in samples:
    x = np.random.normal(0, 30, int(s))

    fx = f(x)
    hx = h(x)
    hax = h_alt(x)

    muf_ = np.mean(fx)
    err1.append(muf_ - muf)
    mud_ = np.mean(fx-hx)
    mudf_ = mud_ + muh
    err2.append(mudf_-muf)
    mud_halt_ = np.mean(fx - hax)
    err3.append(mud_halt_ + mu_halt - muf)
    print(err1[-1], err2[-1])


fig, ax = plt.subplots(1, 1, figsize=(15, 10))



ax.plot(samples, err1, '-', label='Monte Carlo')
ax.plot(samples, err2, '--', label='Monte Carlo + Bad CV')
ax.plot(samples, err3, '-.', label='Monte Carlo + Good CV')
ax.set_xlabel("Samples")
ax.set_ylabel("Error")
ax.set_xscale('log')
ax.legend()
```

we would arrive to this kind of plot:

<div class="d-flex justify-content-center">
    <img src="https://i.imgur.com/tNrKU9g.png"  class="w-100">
</div>

In the figure, the blue line is the standard Monte Carlo estimate with its error behavior with increasing number of samples. Notice that the orange line, the choice where we decide to apply a linear transformation with a slope to $x$, we get relatively bad performance with even worse variance than standard Monte Carlo, but converge nevertheless with enough samples. For the case when we choose a linear transformation with slope 1, things look a lot better, quite early we converge to a very good estimate of the mean, the error drops drastically after a few samples. It might be interesting to look at why is this the case, it seems too good to be true hm... 


Lets put our theory hats on. Here is the game plan, first we'll use a couple of well-known identities with respect to Gaussians in order to write out the variance of the estimator for the case of linear transformations, next we'll compute the optimal slope of the linear transformation.


Starting with the linear transformations, we can already apply the variance identity, where we multiply the variance of $x$ by the slope squared:
<div class="d-flex justify-content-center">
    <img src="https://i.imgur.com/YXiuik6.png"  class="w-75">
</div>

Next up, since the covariance can be written as $\E[f(x)h(x)] - \E[f(x)]\E[h(x)]$,  we write out the first part of the covariance and we notice that we can write it in the compact form because $x$ is normally distributed. 

<div class="d-flex justify-content-center">
    <img src="https://i.imgur.com/LySbDpn.png"  class="w-75">
</div>

Notice that the expectation in the second term is just the mean of the distribution of $x$, which is 0 if we sample from a 0-mean Gaussian. Taking this into consideration, we have all the ingredients written out to inspect the variance of our estimator with control variates. Notice that it turns out that we don't care about the intercept in the linear transformations:

<div class="d-flex justify-content-center">
    <img src="https://i.imgur.com/3yhxDLz.png"  class="w-75">
</div>

Further, we notice that this is a convex function in $a_h$, which is the parameter that we are looking for (scale of $h$), quite simply we can take the derivative of this and set it to 0 to arrive to an optimal slope:

<div class="d-flex justify-content-center">
    <img src="https://i.imgur.com/QjcBfm6.png"  class="w-75">
</div>

Nice, so now we have the optimal slope, which is $a_h = a_f$, now we go back to the original variance term and write $\Var*$ for optimal variance. Plugging the terms back in and realizing that $\Var[x] = \sigma^2$:

<div class="d-flex justify-content-center">
    <img src="https://i.imgur.com/VG2UEAj.png"  class="w-75">
</div>

And voilla! We have reduced our variance to $0$! More concretely, we have shown that for the utterly specific case of $x$ being sampled from a 0 mean Gaussian and $h$ and $f$ being linear transformations. Moreover, the neat thing is that we don't need to care about the intercept. Notice that applying a linear transformation to a Gaussian random variable yields a Gaussian random variable, and we are effectively saying that we don't care what's the mean of $h(x)$. BOOM!