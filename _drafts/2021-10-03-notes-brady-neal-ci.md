---
categories: blog
title: Notes Brady Neal Causal Inference
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


