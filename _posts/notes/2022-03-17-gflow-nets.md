---
title: Generative Flow Networks
layout: post
categories: note
math: true
---

Notes on Generative Flow Networks (GFlowNets), a model class pushed by Yoshua Bengio that enables tractable probabilistic inference and connects MCMC-style sampling with density learning.

Based on a read-through of:
* [Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation](https://openreview.net/pdf?id=Arn2E4IRjEB)
* [GFlowNet Foundations](https://arxiv.org/pdf/2111.09266.pdf)

## Flow Network-Based Generative Models for Diverse Candidate Generation

The goal is to learn a **stochastic policy** capable of sampling multiple high-reward solutions — rather than collapsing to a single mode — such that the probability of sampling a candidate $$x$$ is proportional to its reward $$R(x)$$:

* In standard RL, we seek a policy that maximizes expected return; here, the objective is instead to match a target distribution over terminal states.
* Motivating application: iterative black-box optimization (e.g., drug discovery) where an oracle evaluates rewards for batches of candidates.

Specifically, for terminal states $$x$$, we want:

$$
\pi(x) \approx \frac{R(x)}{Z}
$$

where $$Z$$ is the normalizing constant (partition function) making $$\pi$$ a valid distribution.

Mapping an action sequence to a terminal state is straightforward when the mapping is bijective, but in general multiple trajectories can lead to the same terminal state — forming a **DAG (directed acyclic graph)** over states. Standard autoregressive/policy parameterizations become biased toward states with more incoming paths.

**Viewing the MDP as a Flow Network:** We leverage the DAG structure and learn a flow $$F$$, writing $$F(s, a)$$ for the flow along transition $$s \to s'$$ and $$F(s)$$ for the total flow through state $$s$$. The initial state $$s_0$$ acts as the source with inflow $$Z$$, and each terminal state $$x$$ acts as a sink with outflow $$R(x)$$. Assuming zero reward on internal nodes, flow conservation requires incoming flow to equal outgoing flow at every state $$s'$$:

$$
\sum_{s, a : T(s, a) = s'} F(s, a) = R(s') + \sum_{a' \in \mathcal{A}(s')} F(s', a')
$$

**Proposition 2 (Flow-Induced Policy):** Defining the forward policy as

$$
\pi(a \mid s) = \frac{F(s, a)}{F(s)}
$$

prioritizes actions proportional to their flow. Under flow conservation, the probability of visiting state $$s$$ from $$s_0$$ is

$$
\pi(s) = \frac{F(s)}{F(s_0)},
$$

where the initial flow $$F(s_0) = Z = \sum_{x \in \mathcal{X}} R(x)$$ is the sum of all terminal rewards, and the probability of terminating at $$x$$ is exactly

$$
\pi(x) = \frac{R(x)}{\sum_{x' \in \mathcal{X}} R(x')}.
$$

**Training:** Because flow conservation is recursive, we can train it similarly to TD-learning in deep RL. However, naive squared flow-matching error behaves poorly because flows near the root $$s_0$$ are orders of magnitude larger than flows near the leaves (flows are summed over all downstream paths, not averaged). Taking the logarithm of both sides turns the objective into matching log-flow ratios, with a small smoothing constant $$\epsilon$$ inside the $$\log$$ to prevent $$\log 0$$ and control sensitivity to small vs. large flows.

**Proposition 3:** Shows that training can be **off-policy** — trajectories do not need to be sampled from the current policy $$\pi$$.

**Summary:** GFlowNets provide an interesting framework for sampling diverse candidates proportional to a reward function. In this initial formulation, transitions are restricted to deterministic DAGs and discrete action spaces (continuous action spaces replace the flow-conservation sums with integrals).
