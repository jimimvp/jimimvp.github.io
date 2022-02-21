---
title: What Am I Reading
categories: blog
layout: post
---

Here I will continually update the research papers that I have read, comment them, brainstorm some ideas of improvement.

## Mastering Atari Games with Limited Data (Ye et al. 2021), 2022.02.21 

The authors build on top of MuZero Reanalyze algorithm to introduce EfficientZero, with which they achieve SotA performance on the Atari benchmark. They start by noticing 3 things about the MuZero Reanalyze algorithm:
* lack of supervision of the environment model - because the model is only trained through reward, value and policy functions. Reward can be sparse, giving a sparse signal, value functions are noisy because of **bootstrapping**
* problems with aleatoric uncertainty causes predicted rewards to have large prediction errors.
* off-policy issues of multi-step value - MuZero uses multi-step reward from the environment to learn value functions faster, but we run into off-policy evaluation here, which further hinders performance.

**First** improvement is that the outputs of the representation function and the dynamics function should be the same for $o_{t+1}$, i.e. $s_{t+1} = \hat{s}_{t+1}$. To enforce this, they use SimSiam (cite), a self-supervised algorithm which essentially pulls projected views of the same thing close together. Note that, $s$ here is an internal state representation of the algorithm.

**Second** they tackle the recurrent error accumulation, which they call the "state aliasing problem". A solution to this is predicting the value prefix, rather than using N-step TD for computing the value function target. The intuition about this is kind of cool - humans don't tend to imagine exactly at which step they will receive a reward, but rather have a rough idea, so the value prefix is just going to be a function of the whole state sequence. Note that, this doesn't require bootstrapping and therefore is less noisy, it's trained via direct supervision. In essence though, this doesn't fix the aleatoric uncertainty issue, maybe a distributional RL approach would be of some use here, or ensemble of value prefixes.

**Third** MuZero Reanalyze reuses the data from the replay buffer, which naturally leads to off-policy issues. They propose to re-use the rewards of a dynamic horizon $l$ and re-computing the rest via MCTS for computing the value target. The horizon is smaller the older the data. This causes increased computation cost, but we don't care about this really, because it can be done in parallel.

**In summary**, it is interesting to see what kind of improvements are resulting from these  modifications and they do make sense. However, there can be some improvements. Off the top of my head, using an ensemble of stochastic neural networks such as in PETS or RAZER should be able to deal better with aleatoric and epistemic uncertainty from the environment. Furthermore, I am not really convinced by the off-policy correction technique, it seems a bit ad-hoc since it uses this dynamic horizon of re-computation. Ideas from importance sampling could be used here, but they would introduce more variance to the algorithm - could be interesting to explore, nevertheless.



