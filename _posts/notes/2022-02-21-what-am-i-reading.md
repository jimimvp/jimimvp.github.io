---
title: What I Am Reading
categories: note
layout: post
math: true
---

Running reading notes on research papers — summaries, open questions, and ideas for improvement.

## 5. Mastering Atari Games with Limited Data — EfficientZero (Ye et al., 2021)

The authors build on top of the MuZero Reanalyze algorithm to introduce EfficientZero, achieving state-of-the-art sample efficiency on the Atari 100k benchmark. They start by noticing three issues with MuZero Reanalyze:
* **Lack of supervision of the environment model**: the model is only trained through reward, value, and policy heads. Rewards can be sparse, giving a weak signal, and value targets are noisy because of **bootstrapping**.
* **Aleatoric uncertainty**: causes predicted rewards to have large prediction errors.
* **Off-policy issues with multi-step values**: MuZero uses multi-step rewards from the environment to learn value functions faster, which introduces off-policy evaluation bias.

**First improvement (self-supervised consistency):** The outputs of the representation function and the dynamics function should match for $$o_{t+1}$$, i.e., $$s_{t+1} = \hat{s}_{t+1}$$. To enforce this, they use SimSiam, a self-supervised objective that pulls projected views of the predicted and actual next latent states together.

**Second improvement (value prefix):** To tackle recurrent error accumulation ("state aliasing"), they predict the **value prefix** rather than using $$N$$-step TD for the value target. Humans don't tend to pinpoint the exact step a reward arrives, but rather have a rough estimate over a trajectory segment — so the value prefix is a function of the whole state sequence. Since it doesn't require bootstrapping, it is less noisy and trained via direct supervision. However, this still doesn't fix aleatoric uncertainty; a distributional RL approach or an ensemble of value prefixes might help here.

**Third improvement (model-based off-policy correction):** Reusing data from the replay buffer leads to off-policy issues. They propose reusing observed rewards over a dynamic horizon $$l$$ (shorter for older data) and recomputing the rest via MCTS for the value target. This increases computation cost, though it can be parallelized.

**Implementation technicalities:** Interestingly, the internal representation $$s_t$$ is a small spatial grid (a tiny image), and the forward model taking $$(s_t, a_t)$$ is a ResNet — basically everything is a ResNet except the value prefix head.

**Is the representation good?** When training a decoder on top of the learned representation, EfficientZero produces reasonable decoded frames for future predictions whereas vanilla MuZero decodes mostly noise. Still, neither method clearly encodes every task-relevant detail (for instance, in Atari Breakout the ball is barely visible in decoded frames — possibly because a pixel-wise cross-entropy loss ignores small objects).

**In summary:** These modifications make a lot of sense and yield strong gains. Using an ensemble of stochastic neural networks (as in PETS or RAZER) could better handle aleatoric and epistemic uncertainty. Also, the dynamic-horizon off-policy correction feels somewhat ad-hoc; importance sampling ideas could be cleaner, though they introduce variance.

## 4. Learning and Planning in Complex Action Spaces — Sampled MuZero (Hubert et al., 2021)

Up until this work, MuZero was evaluated in small discrete action spaces that could be enumerated in full during MCTS. Sample-based methods help when scaling to large or continuous action spaces since we don't need to enumerate all actions. The key question is how the sampling procedure interacts with policy improvement and policy evaluation.

In this work, they sample actions and use deterministic transition models. The MuZero algorithm can be understood as alternating policy evaluation and policy improvement, where MCTS provides local policy improvement and the network regresses toward that improved policy.

Questions the paper focuses on:
* How to use the locally improved policy to update the global policy
* How to use it to act
* How to perform an explicit local step of policy evaluation of the improved policy for planning
* How all these steps interact with each other

They separate policy improvement into an `improvement operator` and a `projection operator` (since we approximate the improved policy in the space of realizable policies), and reason about `action-independent` policy improvement operators.

**Sample-based action-independent policy improvement:** We draw finitely many samples from a proposal distribution $$\beta$$ and compute the empirical distribution (supported only on the samples), weighted by $$f(s, a, Z_\beta(s))$$ where $$Z_\beta(s)$$ is a state-dependent normalizing factor. Theorem 4.4 is essentially an application of the CLT showing convergence to the true improvement operator as the number of samples grows.

**Sampled MuZero (Section 5):** Adapts MCTS and the PUCT formula (Silver et al., 2016) to sampled action spaces. Searching naively over sampled actions can be unstable, so they propose using $$\pi_\beta = \frac{\hat{\beta}}{\beta}\pi$$ for action selection — an importance-weighted policy accounting for the proposal distribution $$\beta$$. Interestingly, while the theory motivates importance weighting, in practice they end up discretizing the continuous action space a priori.

## 3. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (Shazeer et al., 2017)

Introduces the Mixture-of-Experts (MoE) layer to scale network parameter count while keeping compute sublinear, since only a subset of experts is evaluated per sample. The MoE layer consists of simple MLPs and a learnable gating network that selects a sparse combination of experts.

All parts are trained jointly end-to-end by backpropagation, with the gating mechanism masking out unselected experts.

**Noisy Top-$$K$$ Gating:** Before taking the softmax, they add tunable Gaussian noise and keep only the top-$$k$$ values, setting the rest to $$-\infty$$. There is no hard-gating gradient estimator like REINFORCE — just softmax weights over the top-$$k$$ experts.

**The shrinking batch problem:** Because $$k < n$$, each expert receives a smaller effective batch. To compensate, they combine data parallelism and model parallelism. An alternative view is to let experts "fight" for data points, effectively pruning unused experts.

**Balancing expert utilization:** To prevent a few early-winning experts from dominating all inputs, they add an auxiliary importance loss penalizing the coefficient of variation ($$\text{std}/\text{mean}$$) of the gating weights across a batch.

**Summary points:** The data-assignment problem is mostly addressed heuristically via load-balancing losses rather than a principled routing mechanism, and stacking multiple MoE layers wasn't deeply explored at the time.

## 2. Adversarially Trained Actor Critic for Offline Reinforcement Learning (Cheng et al., 2022)

A two-player Stackelberg framing of offline RL, where the policy is the "leader" and the critic is the "follower":

$$
\begin{aligned}
&\arg\min_x g(x, y_x) \\
&\text{s.t. } y_x = \arg\min_y h(x, y)
\end{aligned}
$$

where $$y$$ is the follower and $$x$$ is the leader. This generalizes zero-sum min-max formulations (where $$g = -h$$). Applied to RL with critic $$f$$:

$$
\begin{aligned}
&\hat{\pi} = \arg\min_\pi \mathcal{L}(\pi, f^\pi) \\
&\text{s.t. } f^\pi = \arg\min_f \mathcal{L}(\pi, f) + \beta\,\mathcal{E}(\pi, f)
\end{aligned}
$$

Here $$\mathcal{E}$$ is the Bellman consistency loss (standard 1-step TD error, i.e., policy evaluation) and $$\mathcal{L}(\pi, f) = \mathbb{E}[(f(s, \pi) - f(s, a))^2]$$ is the pessimism term (comparing the learned policy's value against dataset actions $$a$$). With this bi-level formulation, ATAC achieves strong results on D4RL, outperforming CQL and COMBO.

* **Robust policy optimization:** Under approximate realizability, the learned policy is guaranteed to be no worse than the behavior policy $$\mu$$ for any $$\beta \ge 0$$, making the algorithm robust to the choice of $$\beta$$.
* **Imitation learning perspective:** When $$\beta = 0$$, the objective reduces to an imitation-learning-like formulation in value space.
* **Practical algorithm (Algorithm 2):** Uses two critic networks updated via projected gradient descent onto $$L_2$$-bounded weights (which enforces bounded complexity as a hard constraint, unlike $$L_2$$ regularization).
* **Double Q Residual Algorithm (DQRA) loss:** Mitigates the deadly triad (function approximation, bootstrapping, off-policy updates) using a convex combination of Bellman consistency losses between the minimum of the two critics and each critic itself.
* **Policy projection & loss:** Policies are projected onto a class with a minimum entropy constraint, and the policy optimizes only a single critic (taking the minimum of two critics causes instability for small $$\beta$$).

## 1. Lessons from AlphaZero for Optimal, Model Predictive and Adaptive Control (Bertsekas, 2021)

Dimitri Bertsekas' perspective on AlphaZero. In the abstract, he highlights the off-policy vs. online gap: the online player plays significantly better than the extensively trained offline policy due to long-lookahead minimization, which corrects for imperfections in the offline value approximation:

> An important lesson from AlphaZero and TD-Gammon is that performance of an off-line trained controller can be greatly improved by on-line approximation in value space, with long lookahead (whether involving minimization or rollout with an off-line obtained policy).

Approximation in value space amounts to a step of Newton's method for solving the Bellman equation, where the starting point is provided by offline training.
