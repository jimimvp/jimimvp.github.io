---
categories: note
title: Credit Assignment in Reinforcement Learning
layout: post
---

The problem of credit assignment is a long-lasting issue in reinforcement learning. In short, it's about which action is actually "caused" the reward in the future. Here I am looking at two papers that address this problem.

* Synthetic Returns for Long-Term Credit Assignment 

# Synthetic Returns for Long-Term Credit Assignment (24th Feb 2021)

1-step TD error approach to learning suffers with delayed rewards with long periods or when between the action and reward there are uncontrollable events that contribute.

Their proposed method (optimizing synthetic returns, explanation will come later) combined with the IMPALA agent is able to solve the skiing game in Atari, which is difficult because of the delayed reward, 25 times faster than the state of the art of the time.

**State-associative learning.** The goal is to learn associations between pairs of states such that the state that was visitied prior to the other one is predictive of the reward and the future state (I think a better word here would be future timestep). This enables the algorithm to construct a synthetic return that connects the reward to the previous state by skipping timesteps.

**Door-key example.** If you have a door that you need to open to receive reward, you first need to pick up the key. So the credit of the reward shold be assigned to the state (transition) before (pickiping up the key), otherwise with one-step TD error we are going to obtain no reward signal for the key pickup transition and the value function approximator has a hard time of figuring this out, a lot of samples are needed.

**Equation 1., the reward prediction loss.** This is a bit weird to me, since the gating function $$g(s_t)$$ gates (softly) the sum over the outputs of $$c(s_k)$$ for T previous states and we have the baseline subtracted. So the prediction of the reward is in fact the sum of the state predictions. What is also a bit weird is that this is the internal agent state and not the actual environment observation (since partially observable). That should mean that the state $$s_{t-1}$$ of the previous states contains a summary of the history, should be predictive of the reward at $$s_t$$. There is no notion of causality here, i.e. which transition actually caused the reward. In the end, the augmented reward that they use is
$$ \hat{r}_t = \alpha c(s_t) + \beta r_t
$$

Apparently there is a `JAX implementation of this`, yuppee!

Alright, a bit later after the equation they explain why they are summing over $$c(s_k)$$, the motivation is that the reward prediction is cast as a regression problem with weights, so each of the past states has a certain "weight" that is associated with their prediction, meaning how relevant is their prediction. It must have something to do with the architecture of the reward predictor, it's still somewhat ad-hoc to me since there are no regression weights in the sum.

**Synthetic return.** This is what they call $$c(s_k)$$.

Further, they assume that the contribution of past states to a future state reward is `sparse`, that's why the gating mechanism is introduced before the sum. I think it probably didn't work without the gating mechanism, because there is no causal connection here between past action and reward, `this might be a problem worth exploring`.

Notice that the baseline in `Eq. 1` depends on $$s_t$$ and is subtracted, the argument here is that $$b(s_t)$$ can still overtake the prediction of the reward in spite of the previous states, but I don't see any way how it is encouraged to do so actually. Apparently, re-using $$c(\cdot)$$ didn't work as well as using a separate function for this.

The $$\alpha$$ and $$\beta$$ hyperparameters are there because of the trade-off between this useful inductive bias that 1-step TD error has (that temporal recency is a good indicator for reward attribution) and this synthetic return. `Are there any nice properties of this in the tabular case actually?`.

**Experiments.** I am pretty much skipping this section in interest of time, long story short - the algorithm is very good in cases of long delayed rewards. The only thing that is peculiar to me that the objective for the synthetic returns is modified.

**Summary/Discussion** Do I don't get it here why are they calling it a synthetic return instead of a synthetic reward, which it is. They mention that they cannot offer convergence guarantees because of the use of multiplicative gates and because of the use of an additive regression model they cannot offer guarantees for optimal credit assignment :), but the cool things is that they propose SA variants that overcome this in the appendix. Another problem is that this method has high variance across seeds for some reason.


**BONUS: Appendix.** In the current formulation the reward contributions of the summation might be double counted, so there is no reward conservation. The section 6.4 limitation of additive regression I am not getting exactly, now suddenly we have the function $$c(\cdot, \cdot)$$, I need to read this more thoroughly.











