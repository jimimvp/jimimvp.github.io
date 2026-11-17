---
title: Variance Reduction via Control Variates
categories: blog
layout: math_post
---

Control variates are one of the most classical and effective ways to reduce the variance of Monte Carlo (MC) estimates. Consider the setup where we want to estimate $$\mu_f = \E[f(x)]$$ with MC. Imagine that we have access to a correlated auxiliary function $$h(x)$$ with known expectation $$\mu_h = \E[h(x)]$$. We can then consider the difference $$d(x) = f(x) - h(x)$$ and estimate its expectation via MC:

$$
\hat{\mu}_d = \frac{1}{n}\sum_{i=1}^n \big(f(x_i) - h(x_i)\big)
$$

First question to ask: is adding $$\mu_h$$ back to $$\hat{\mu}_d$$ an unbiased estimator of $$\mu_f$$? Yes — by linearity of expectation:

$$
\E[\hat{\mu}_d + \mu_h] = \E[f(x) - h(x)] + \E[h(x)] = \E[f(x)] = \mu_f.
$$

So we define our control-variate estimator of $$\mu_f$$ as:

$$
\hat{\mu}_f = \hat{\mu}_d + \mu_h.
$$

Next, and this is the far more interesting question: how does the variance of this estimator behave? Using the standard variance identity $$\Var[\hat{\mu}_d] = \E[\hat{\mu}_d^2] - \mu_d^2$$ for $$n$$ i.i.d. samples (and noting that $$\mu_h$$ is a deterministic constant with zero variance):

$$
\Var[\hat{\mu}_f] = \Var[\hat{\mu}_d] = \frac{1}{n}\Var[f(x) - h(x)].
$$

Note that even when we don't have a closed-form expression for $$\mu_h$$, it can still be beneficial to estimate $$\mu_h$$ separately (for example using a much larger or cheaper sample budget) if the variance of that estimator is smaller than estimating $$\mu_f$$ directly.

## When Does a Control Variate Actually Help?

Let's expand the variance of the difference $$f(x) - h(x)$$:

$$
\Var[f(x) - h(x)] = \Var[f(x)] + \Var[h(x)] - 2\,\Cov[f(x), h(x)].
$$

This immediately tells us two things:
1. The variance grows with the individual variances of $$f(x)$$ and $$h(x)$$.
2. We only achieve variance reduction ($$\Var[f(x) - h(x)] < \Var[f(x)]$$) when $$2\,\Cov[f(x), h(x)] > \Var[h(x)]$$.

In other words, a poorly scaled or weakly correlated $$h(x)$$ can actually *increase* variance rather than reduce it!

### Optimal Scaling for Any Control Variate

To guarantee we never make the variance worse, we can introduce a scalar multiplier $$\beta \in \R$$ and form the estimator

$$
\hat{\mu}_f(\beta) = \frac{1}{n}\sum_{i=1}^n \Big(f(x_i) - \beta\big(h(x_i) - \mu_h\big)\Big).
$$

This remains unbiased for *any* $$\beta \in \R$$. Its per-sample variance is a convex quadratic in $$\beta$$:

$$
\Var\big[f(x) - \beta(h(x) - \mu_h)\big] = \Var[f(x)] + \beta^2\,\Var[h(x)] - 2\beta\,\Cov[f(x), h(x)].
$$

Differentiating with respect to $$\beta$$ and setting the derivative to zero gives the optimal scaling factor:

$$
\beta^* = \frac{\Cov[f(x), h(x)]}{\Var[h(x)]}.
$$

Plugging $$\beta^*$$ back in reveals a clean geometric interpretation in terms of the Pearson correlation $$\rho_{f, h} = \frac{\Cov[f(x), h(x)]}{\sqrt{\Var[f(x)]\Var[h(x)]}}$$:

$$
\Var^* = \big(1 - \rho_{f, h}^2\big)\,\Var[f(x)].
$$

Even if $$\beta^*$$ isn't known analytically, it can be estimated from samples (or learned via gradient descent on the empirical variance when $$h_\theta(x)$$ is a parametric family such as a neural network). The closer $$\lvert \rho_{f, h} \rvert$$ is to $$1$$, the closer the estimator variance gets to $$0$$.

## A Concrete Python Example

To make this concrete, suppose we want to estimate $$\E[f(x)]$$ where $$f(x) = x - 10$$ and $$x \sim \gN(0, 30^2)$$. The true mean is $$\mu_f = -10$$, and we have access to samples $$x$$ from this zero-mean Gaussian. In the code below, we compare 3 estimators:
1. Pure Monte Carlo
2. Monte Carlo with a poorly scaled control variate $$h(x) = 10x - 2$$
3. Monte Carlo with a well-matched control variate $$h(x) = x - 5$$

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
    mud_ = np.mean(fx - hx)
    mudf_ = mud_ + muh
    err2.append(mudf_ - muf)
    mud_halt_ = np.mean(fx - hax)
    err3.append(mud_halt_ + mu_halt - muf)

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
ax.plot(samples, err1, '-', label='Monte Carlo')
ax.plot(samples, err2, '--', label='Monte Carlo + Bad CV')
ax.plot(samples, err3, '-.', label='Monte Carlo + Good CV')
ax.set_xlabel("Samples")
ax.set_ylabel("Error")
ax.set_xscale('log')
ax.legend()
```

We arrive at the following error plot:

<div class="d-flex justify-content-center">
    <img src="https://i.imgur.com/tNrKU9g.png" alt="Monte Carlo vs. Bad and Good Control Variates error convergence plot" class="w-100">
</div>

In the figure, the blue line is the standard Monte Carlo estimate and its error behavior as the number of samples increases. Notice that the orange line — where we use a control variate with slope $$10$$ — gives worse variance than standard Monte Carlo, though it still converges with enough samples. For the green line, where we choose a linear transformation with slope $$1$$, the error drops to zero immediately. Let's verify why this happens algebraically.

## Checking the Linear Gaussian Case

Let $$f(x) = a_f x + b_f$$ and $$h(x) = a_h x + b_h$$ with $$x \sim \gN(0, \sigma^2)$$. Starting with the variance of linear transformations, we multiply the variance of $$x$$ by the slope squared:

$$
\Var[f(x)] = a_f^2\,\Var[x], \qquad \Var[h(x)] = a_h^2\,\Var[x].
$$

Next, since the covariance is $$\Cov[f(x), h(x)] = \E[f(x)h(x)] - \E[f(x)]\E[h(x)]$$ and $$\E[x] = 0$$, the cross-terms vanish and we get:

$$
\Cov[f(x), h(x)] = \E[(a_f x + b_f)(a_h x + b_h)] - b_f b_h = a_f a_h \E[x^2] = a_f a_h\,\Var[x].
$$

Putting these ingredients together gives the variance of the difference. Notice that the intercepts $$b_f$$ and $$b_h$$ drop out completely:

$$
\Var[f(x) - h(x)] = (a_f^2 + a_h^2 - 2 a_f a_h)\,\Var[x] = (a_f - a_h)^2\,\Var[x].
$$

In our "Bad CV" example, $$a_f = 1$$ and $$a_h = 10$$, so $$(a_f - a_h)^2 = 81$$ — inflating the variance by $$81\times$$ relative to plain MC! Setting the derivative with respect to $$a_h$$ to $$0$$ gives the optimal slope:

$$
\frac{\partial}{\partial a_h}\Var[f(x) - h(x)] = 2(a_h - a_f)\,\Var[x] = 0 \implies a_h^* = a_f.
$$

Plugging $$a_h^* = a_f$$ back into the variance with $$\Var[x] = \sigma^2$$:

$$
\Var^*[f(x) - h(x)] = (a_f - a_f)^2 \sigma^2 = 0.
$$

And voilà — we have reduced our variance to $$0$$! In practice, this exact principle underpins **baselines in policy gradient methods** (REINFORCE / actor-critic, where $$\E_{a \sim \pi}[\nabla_\theta \log \pi_\theta(a \mid s)\, b(s)] = 0$$ makes $$b(s)\nabla_\theta \log \pi_\theta$$ a zero-mean control variate) as well as **score-function gradient estimators** like REBAR and LAX for discrete latent variables.
