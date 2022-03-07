---
title: Reinforcement Learning Roundup (1)
categories: note
layout: post
---

* A Convergent and Efficient Deep Q Network Algorithm
* Offline Reinforcement Learning with Soft Behavior Regularization


# A Convergent and Efficient Deep Q Network Algorithm

Generally, temporal difference learning does not guarantee convergence when function approximation is used
Modify the loss of DQN so as to make it non-increasing.

It is clear that the TD error with the target that shares parameters with the current estimate may not converge, because both the estimate and the target move at the same time.

Introduction of the target network is not well-principled and it doesn't fully resolve the divergence issue. DQN requires a lot of hyperparameter tuning in order to achieve good performance.
Previous work tried to update $$\theta$$ in a direction that is perpendicular to the gradient of the maximum of Q, however this has shown not to work for the general case.

**Mean Squared Bellman Error**. I don't quite get what is the difference between this error and the one-step TD error used in DQN? The fixed dataset? The fact that we are not using target value functions but rather the same vlaue function for computing the TD error?

**Interesting fact**. The interesting fact about this loss in the paper that they show is that the Hessian with respect to the Q-values is ill-conditioned. The condition number of the Hessian grows with squared complexity with the number of transitions, the condition number being $$\kappa = \frac{\lambda_{max}}{\lambda_{min}} $$ where $$\lambda$$ are the eigenvalues of the Hessian. In practice, $$\kappa$$ is usually $$10^3-10^4$$!

**Tendency of maintaining the average prediction.**  The squared Bellmann error because of its formulation has the problem of maintaining the average prediction between the Q values in time step t and t+1 of the transition. This means that learning efficiency is bad for $$\gamma < 1$$ and for $$\gamma = 1$$ it doesn't learn at all. (very handwavy written here **TODO**).

**DQN as fitted value iteration.** So what is happening here... We update the target to be the argmin of the DQN loss. This is different than the standard way of slowly updating the target via polyak averaging as an example, it's really interesting that this modification leads to better properties with respect to convergence. In practice though, the minimum is found by stochastic gradient descent.


**Final loss.** In the end, they take the maximum over the MSBE and DQN loss as the loss function. The C-DQN loss is convergent, meaning that the iteration (updating parameters with gradient) is both bounded from below and non-increasing. In practice it's found the the MSBE is reduced much faster than the DQN error so the C-DQN loss focues primarily on the DQN error. Would be interesting to see how does this fare in the distributional RL setting.

**Remark.** Interestingly, the modficiation that the ATAC paper did to the loss introduced in this paper is simply to do a convex combination of the MSBE and DQN loss, but the parameter that they use for the convex combination is 0.5, meaning it's just scaling the learning rate in the end? I am confused a bit here.


**Potential research directions.** Extensions of C-DQN to stochastic transitions without bias. C-DQN in the distributional RL context. The loss of C-DQN is non-smmooth, it is unclear how this affects performance and if something can be done about it. Are there better strategies to select samples from the experience? Since some of the samples contribute more to one part of the loss (MSBE) vs. the other (DQN loss).


* Do we have a good explanation of why the condition number of the Hessian affects the gradient-based learning so much? I have it roughly, but something more concrete would be better.


# Offline Reinforcement Learning with Soft Behavior Regularization

The paper proposes ways to deal with the following problems in offline RL:
* lack of policy improvement guarantee.
* hard to estimate Q because of distribution shift.
* too much conservativism due to state-independent regularization weight.

Lemma 1 tells us how to have guaranteed performance improvement with respect to the behavior policy. Roughly, if we have access to samples from the current policy to estimate the expectation, we can maximize the advantage function of the behavior policy. It's relatively straight-forward why this leads to guaranteed policy improvement. The problem is that we don't have access to sampled transitions from $$\pi$$ in the offline setting.

One way to deal with this is a trust-region method, i.e. assume that the behavior policy and the current policy have similar state visitation distribution if they are close to each other. What Eq. 4 does is introduces a total variation distance penalty (with some multiplication factors that pop out of the theory of previous papers) to the objective of Lemma 1, i.e. penalizing getting too far from the behavior policy. The problem here is that, using this naively will be too restrictive and constraining the single action distributions might not be enough to ensure the trust region property. **Instead** they do directly importance sampling, i.e. calculating the state visitation density ration and multiplying the advantage.

Remaining question - how to estimate this density ratio from offline data? Theorem 1 result is pretty neat, it tells  under which conditions is the density ratio uncoverable, i.e. when it satisfies a Bellmann-like equation, where a discrepency measure between distributions is used. In this paper, they make use of a kernel method MMD(Maximum Mean Discrepency).

In order to satisfy the condition that $$\pi$$ lies in the support of $$\mu$$ they introduce a log-barrier constraint to the optimization problem, i.e. the log of the behavior policy should be above a certain $$\epsilon$$. They relax the hard-constraint and use it just as a penalty term.

Interesting things about their proposed objective:
* They don't need to estimate the value of the current policy, but rather of the behavior policy, so they don't suffer in terms of overestimation bias.
* By not using the KL divergence, there is no entropy term of the current policy that gets increased. They argue that this is only useful in the online case where we have the problem of exploration.
* By using a state-dependent regularization weight (density ratio / importance weight) the regularization is high when the state visiation density of the policy is small vs. the behavior policy. In the end what this means that the log-barrier term overtakes the loss and we end up with maximizing the likelihood of the behavior policy.


**Question to the paper.** Choice of kernel is important here, I can't imagine that this density estimation works well (in image-based domains it won't work at all). One limitation of SBAC is that the behavior policy needs to be estimated, multiple problems can arise here (we need an expressive estimator, and the data may come from **multiple** policies!).


