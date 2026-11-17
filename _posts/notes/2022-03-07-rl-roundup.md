---
title: Reinforcement Learning Roundup (1)
categories: note
layout: post
math: true
---

Notes on two reinforcement learning papers:
* A Convergent and Efficient Deep Q Network Algorithm (C-DQN)
* Offline Reinforcement Learning with Soft Behavior Regularization (SBAC)

## A Convergent and Efficient Deep Q Network Algorithm

Generally, temporal difference (TD) learning does not guarantee convergence when function approximation is used. This paper modifies the DQN loss so that it is non-increasing.

The TD error with a target that shares parameters with the current estimate may diverge, because both the estimate and the target move at the same time. Introducing a separate target network helps in practice, but is not fully principled and doesn't completely resolve divergence; DQN still requires substantial hyperparameter tuning.

**Mean Squared Bellman Error (MSBE):** The paper highlights that the Hessian of the MSBE with respect to the $$Q$$-values is severely ill-conditioned. Its condition number $$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$ grows quadratically with the number of transitions, reaching $$\kappa \sim 10^3\text{--}10^4$$ in practice.

**Tendency to maintain the average prediction:** Because of its formulation, the MSBE tends to preserve the average prediction between $$Q(s_t, a_t)$$ and $$Q(s_{t+1}, a_{t+1})$$ rather than propagating values forward cleanly, which slows learning for $$\gamma < 1$$ and stalls learning when $$\gamma = 1$$.

**DQN as fitted value iteration:** Here the target is updated toward the $$\arg\min$$ of the DQN loss (approximated via SGD), rather than slowly updated via Polyak averaging, which yields better convergence properties.

**Final loss:** They take the maximum over the MSBE and the DQN loss as the objective. The resulting C-DQN loss is convergent — parameter updates are both bounded from below and non-increasing. In practice, MSBE shrinks much faster than the DQN error, so C-DQN primarily tracks the DQN error.

**Remark:** Interestingly, the DQRA modification in the ATAC paper uses a convex combination of MSBE and the DQN loss (with weight $$0.5$$), which behaves differently from taking the max.

**Potential research directions:**
* Extending C-DQN to stochastic transitions without bias, and to distributional RL.
* The C-DQN loss is non-smooth (due to the $$\max$$); it is worth studying how this affects optimization and whether sample selection strategies help since different samples contribute differently to MSBE vs. the DQN loss.

## Offline Reinforcement Learning with Soft Behavior Regularization

This paper addresses three core challenges in offline RL:
* Lack of a policy improvement guarantee
* Difficulty estimating $$Q$$ under distribution shift
* Over-conservatism due to state-independent regularization weights

Lemma 1 establishes guaranteed performance improvement over the behavior policy $$\mu$$: if we can estimate expectations under the current policy $$\pi$$, maximizing the advantage of the behavior policy guarantees improvement. The catch in the offline setting is that we don't have access to transitions sampled from $$\pi$$.

One standard remedy is a trust-region penalty (e.g., total variation distance), which can be overly restrictive when applied uniformly across states. **Instead**, they apply importance sampling using the state-visitation density ratio to weight the advantage.

**Estimating the density ratio:** Theorem 1 characterizes when the density ratio is recoverable from offline data — namely when it satisfies a Bellman-like equation under a distributional discrepancy measure, which they instantiate with kernel Maximum Mean Discrepancy (MMD).

To keep $$\pi$$ within the support of $$\mu$$, they add a log-barrier constraint ($$\log \mu(a \mid s) \ge \epsilon$$), relaxed into a penalty term.

Key properties of the resulting objective:
* They only need to evaluate the value of the behavior policy $$\mu$$ rather than $$\pi$$, avoiding overestimation bias.
* By avoiding a KL penalty, there is no entropy bonus on $$\pi$$ (which they argue is mainly useful for online exploration).
* Using a state-dependent regularization weight (the state-visitation density ratio) increases regularization where $$\pi$$ deviates most from $$\mu$$'s visitation density, causing the log-barrier to dominate and fall back toward maximizing behavior likelihood.

**Open questions:** The choice of kernel for MMD is critical — it is unclear how well this density-ratio estimation scales to high-dimensional or image-based domains. Additionally, SBAC requires estimating the behavior policy $$\mu$$, which can be challenging when offline data comes from **multiple** behavior policies.
