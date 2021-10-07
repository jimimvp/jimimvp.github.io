---
categories: blog
title: Notes Brady Neal Causal Inference
layout: post
---

In this post 

## Simpson's Paradox

Covid example, we have treatment, condition and outcome.

Simpson's paradox is the basically flipping of a statistic when we condition on something vs. looking at the marginals. 
This is the result of unequal weighting between different situations.

The question remains is which treatment should we choose, causal inference comes here into play.
The condition might be a confounder.



## Correlation does not imply causation

Causal association instead of causal link!


## Potential Outcomes Framework

The causal effect is basically the difference between the outcome when applying the treatment vs outcome without applying the treatment.
Counterfactuals are missing data, since we only have "factual" information.

Average treatment effect - taking an expectation over the outcomes (sampling individuals).
This amounts to difference in conditional expectations.

$$
\begin{equation}
    \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)]  \neq \mathbb{E}[Y| T=1] -  \mathbb{E}[Y| T=0]
\end{equation}
$$

It is not equal is because association is not causation.

The groups are not comparable, because mostly drunk people sleep with shoes, are more likely to take the pill.


What assumptions can we make to make the average treatment effect equal to the causal effect?
Ignorability: (Y(1), Y(0)) are independent of treatment T.

Exchangeability: if we exchange the treatment groups 1 and 0, then the expected values are the same.

Identifiability: going from causal quantities to statistical quantities. A causal quantity is identifiable if we can compute it from a purely statistical quantity. Ignorability and exchangeability are assumptions that allow us to do this. Randomized controlled trials are a mechanism through which we can achieve this, such that the groups are comparable.

Graphical interpretation: we remove the confounding arrow by doing an RCT. 

Conditional exchangeability, conditional average treatment effect.
Going from this to the average treatment effect involves marginalizing over the conditioning variable, this is also called the **adjustment** formula.
CE and CI are also called unconfoundedness, and this assumption cannot be tested.

Positivity: for all values of covariates x that are present in the population we have that the probability of treatment is greater than 0 for all values of treatment. This would result in division by 0 if we do not assume positivity. Model fitting extrapolation is also an issue.

No interference assumption: ?

Consistency assumption: if the treatment T = t is made, then the outcome is Y = Y(t). This can be rephrased such that we don't have different versions of the treatment.


## Tying Everything Together


No interference assumption.


# Graphs

## Bayesian Networks

Modelling the joint distribution, ie statistical modelling - no causality.

Local markov assumption: given its parents in the DAG, a node X is independent of all of its non-descendants.

Bayesian network factorization, each factor is just the probability conditioned on its parents.

Minimality assumption: 
    1. local markov assumption. This by itself is not enough because we can represent one joint distribution in many different graphs
    2. lets us read out statistical dependencies.


## Causality

A variable X is said to be a cause of a variable Y if Y can change in response to changes in X.

Causal edges assumption - every parent is a direct cause of all of its children, this allows us to talk about causal dependencies.

causal edges assumption subsumes the second part of the minimality assumptions.


### Chains

### Forks

### Colliders (Immoralities)

Colliders are special, because when we condition on the collider, we open up the path. In fact, if we condition on descendants of the collider, we also unblock the path. Intuitively, if we set a value for the collider, then to achieve this value we need specific combinations of the parents in order to achieve it. This relation transfer to the descendants of the collider obviously. 

blocked paths vs. unblocked paths

d-separation - two sets of nodes are d-separated by a set of nodes Z if all paths between any node of X and any node of Y are blocked.

given the local markov assumption, d-separation implies conditional independence (global Markov assumption), commonly also just referred to as the markov assumption.


# Causal Models


## The do-operator

Making the distinction between conditioning and interveneing.
Conditioning means that we are just restricting the data to a specific subset of the data.
In contrast, intervention is acting for the whole population. This is quite a philosopchical context. 

p(Y(t) = y) = p(Y=y | do(T=t)) is just an interventional distribution, this connects directly to potential outcomes.

Average treatment effect is just a difference between the first moments of two interventional distributions.
common observational distribution P(Y,T, X), interventional P(Y | do(T = t)), this is different to P(Y | T = t).
Interventional quantities might require experiments.

Identifiability: We want to take a causal estimand and remove the do operator to arrive to a statistical estimand.

## Main assumption: modularity

A casual mechanism for a specific variable in the graph is all of the parents of a variable and their arrows to the variable.

Modularity assumption - if we intervene on a node x_i, then the only mechanism that changes is x_i. In other words, the causal mechanisms are **modular**, this is also called independent mechanisms, autonomy, invariance. I hate the independence condition.
## Backdoor adjustment


Truncated factorization is a consequence of the modularity assumption. Changes the product over i to a product over nodes that are not in the intervention set.

## Structural Causal Models

