---
title: Generalized Linear Models
categories: blog
layout: post
---

# Generalized Linear Models

We need:
* Link function that maps the latent function $$f_x$$ to the non-Gaussian likelihood $$p(y | f_x)$$, example is the logistic function for binary classification.
* Question: how do we optimize this efficiently? Since the likelihood is not Gaussian anymore, but we have a Gaussian prior, we cannot nicely compute it in closed form by using nice properties of Gaussians, the posterior is not Gaussian. Luckily, we can turn the posterior into a Gaussian (locally) by applying Laplace approximation.
* Laplace approximation: we compute approximately the maximum of the posterior (we can achieve this by any optimization method). The maximum becomes the mean of our Gaussian approximation. Next we compute the Hessian, i.e. we compute a quadratic approximation of the posterior at the maximum. The Hessian is the negative inverse of the covariance matrix.


