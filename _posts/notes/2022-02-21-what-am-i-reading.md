---
title: What Am I Reading
categories: blog
layout: post
---

Here I will continually update the research papers that I have read, comment them, brainstorm some ideas of improvement.

## Learning and Planning in Complex Action Spaces - MuZero for Continuous Spaces (Hubert et al., 2021), 2022.02.21

Up until now, MuZero was evaluated in simple action spaces, that are so small that they could be enumerated in full through MCTS.
Sample-based methods show benefits when we shift to large complex spaces, since we don't need to enumerate all of the actions (example, continuous spaces) (but we need to pay something here for our sample-based estimate?). Key question: how does the sampling procedure interact with policy improvement and policy evaluation?

In this work, they sample actions and use deterministic transition models.

The MuZero algorithm can be understood as the combination of policy evaluation and policy improvement. The MCTS step provides policy improvement in form of regressing the policy towards a better policy extracted locally. (come to think of it, off-policy, bootstrapping, approximation - deadly triad?). 

Questions that the paper focuses on (as per the authors):
• how to use the locally improved policy to learn about
the global policy
• how to use it to act
• how to perform an explicit local step of policy evalua-
tion of the improved policy for planning
• how all these steps interact with each other

So far in the paper, it's very abstract... Which problem do they actually solve, how to select actions for policy improvement? The authors say that they propose a framework to reason about the sampled policy improvement by re-writing the expectations with respect to the improved policy. Apparently this is achieved by looking at `action-independent` policy improvement operators.

They separate the process of policy improvement into an `improvement operator` and `projection operator`. Why projection? Because we approximate the improved policy in the space of realizable policies.

**Sample-based action-independent policy improvement** well, basically we sample from a proposal distribution $\beta$ finitely many samples and compute the empirical distribution (non-zero only on the samples), this is multiplied by $f(s,a, Z_\beta(s))$ where the last term is state-dependent normalizing factor.

At first glance, the theorem in 4.4 seems to a be a straight-forward application of the CLT, to show that with infinite number of samples we converge to the true improvement operator in distribution.


**Section 5, finally, we get to Sampled MuZero** so this part is dealing with how to adapt MCTS, more concretely PUCT formula (Silver et al., 2016) in order to use it in sampled action spaces. One straight-forward approach is search over sampled actions, and keep everything unchanged. This can though lead to unstable results. (it's still unclear to me how the sampling is done here, for each step separately, or we just sample finitely many actions from action space and continue business as usual?). What the authors rather propose is using $\pi_\beta$ for action selection, where $\pi_\beta = (\frac{\hat \beta}{\beta}\pi$, i.e. an importance weighted policy since the sampling was done from the proposal distribution $\beta$. 

**this paper needs more attention!**



## Mastering Atari Games with Limited Data (Ye et al. 2021), 2022.02.21 

The authors build on top of MuZero Reanalyze algorithm to introduce EfficientZero, with which they achieve SotA performance on the Atari benchmark. They start by noticing 3 things about the MuZero Reanalyze algorithm:
* lack of supervision of the environment model - because the model is only trained through reward, value and policy functions. Reward can be sparse, giving a sparse signal, value functions are noisy because of **bootstrapping**
* problems with aleatoric uncertainty causes predicted rewards to have large prediction errors.
* off-policy issues of multi-step value - MuZero uses multi-step reward from the environment to learn value functions faster, but we run into off-policy evaluation here, which further hinders performance.

**First** improvement is that the outputs of the representation function and the dynamics function should be the same for $o_{t+1}$, i.e. $s_{t+1} = \hat{s}_{t+1}$. To enforce this, they use SimSiam (cite), a self-supervised algorithm which essentially pulls projected views of the same thing close together. Note that, $s$ here is an internal state representation of the algorithm.

**Second** they tackle the recurrent error accumulation, which they call the "state aliasing problem". A solution to this is predicting the value prefix, rather than using N-step TD for computing the value function target. The intuition about this is kind of cool - humans don't tend to imagine exactly at which step they will receive a reward, but rather have a rough idea, so the value prefix is just going to be a function of the whole state sequence. Note that, this doesn't require bootstrapping and therefore is less noisy, it's trained via direct supervision. In essence though, this doesn't fix the aleatoric uncertainty issue, maybe a distributional RL approach would be of some use here, or ensemble of value prefixes.

**Third** MuZero Reanalyze reuses the data from the replay buffer, which naturally leads to off-policy issues. They propose to re-use the rewards of a dynamic horizon $l$ and re-computing the rest via MCTS for computing the value target. The horizon is smaller the older the data. This causes increased computation cost, but we don't care about this really, because it can be done in parallel.

**In summary**, it is interesting to see what kind of improvements are resulting from these  modifications and they do make sense. However, there can be some improvements. Off the top of my head, using an ensemble of stochastic neural networks such as in PETS or RAZER should be able to deal better with aleatoric and epistemic uncertainty from the environment. Furthermore, I am not really convinced by the off-policy correction technique, it seems a bit ad-hoc since it uses this dynamic horizon of re-computation. Ideas from importance sampling could be used here, but they would introduce more variance to the algorithm - could be interesting to explore, nevertheless.



