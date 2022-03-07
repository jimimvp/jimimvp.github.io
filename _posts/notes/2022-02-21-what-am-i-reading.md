---
title: What I Am Reading
categories: note
layout: post
---

Here I will continually update the research papers that I have read, comment them, brainstorm some ideas of improvement.

## Lessons from AlphaZero for Optimal, Model Predictive and Adaptive Control ([Link](https://arxiv.org/pdf/2108.10315.pdf))

Dimitri Bertsekas' take on the alpha-zero craze.
In the abstract, he starts directly with highlighting the off-policy fact, that the off-line player is not used directly online in order to obtain further samples. The online player plays better than the extensively trained offline player, apparently due to long lookahead minimization which corrects for imperfections of the offline player. More concretely:

> An important lesson from AlphaZero and TD-Gammon is that performance of an off-line trained controller can be greatly improved by on-line approximation in value space, with long lookahead (whether involving minimization or rollout with an off-line obtained policy).

The approximation in value space amounts to a step of Newton's method for solving the Bellman equation, the starting point of the Newton step is based on results of offline training. I'm curious how this is argued, Newton's method uses curvature information (Hessian) in order to scale the gradient appropriately.

The key question of the paper is why are there beneficial effects of online decision making on top of offline training?



## Adversarially Trained Actor Critic for Offline Reinforcement Learning ([Link](https://arxiv.org/pdf/2202.02446.pdf), rough pass)

Two-player Stackelberg framing of offline RL, where the policy is the "leader" and the critic is the follower.
More concretely, a two-player Stackelberg game has the formulation

$$\begin{eqnarray}
\arg\min_x g(x, y_x) \nonumber \\
s.t. y_x = \arg\min_y h(x, y) \nonumber
\end{eqnarray}$$

Where y is the follower and x is the leader. 
The Stackelberg game can be seen as generalizing previous min-max zero-sum formulations, which are of the same form, but $$g = -h$$.
The application to RL is pretty much straight-forward, if we take the critic being the function $$f$$, then the formulation becomes the following

$$\begin{eqnarray}
\hat\pi = \arg\min_\pi \mathcal L(\pi, f^\pi) \nonumber \\
s.t. f^\pi = \arg\min_f \mathcal L(\pi, f) + \beta \mathcal E(\pi, f) \nonumber
\end{eqnarray}$$

Where the last term $$\mathcal E$$ for the constraint is the Bellman consistency loss which is basically your standard 1-step TD-error when applying the Bellman operator, i.e. it's policy evaluation.
The first the is the pessimism term which is $$L(\pi, f) = \mathbb E[(f(s,\pi)-f(s,a)^2]$$, unfortunately at this point in the paper it isn't clear to me where $$a$$ is drawn from, I suppose the offline dataset.

With the by-level optimization formulation, the algorithm achieves state-of-the-art in D4RL, beating algorithms such as CQL, COMBO.

**Robust policy optimization** the interesting theoretical result that the authors show is that with this bi-level optimization formulation the learned policy is guaranteed to be no worse that the behavior policy $$\mu$$ for any $$\beta \ge 0$$. Furthermore, the algorithm is robust to the choice of the hyperparameter $$\beta$$. This holds under the approximate realizability assumption (which basically states that the achievable critic approximate is sufficiently close to the actual target), which is weaker than

**Proposition 3 performance difference lemma.** Here is a useful result in the proof that the policy is no-worse than the behavior policy.

**Imitation learning perspective.** The authors state that there is a clear connection between this formulation and imitation learning in that the objective resembles imiation learning for the case of $$\beta=0$$. This is interesting, because all of the objectives in the optimization problem are in value space, why is this exactly imitation learning with matching state occupancy? 


**Theoretical algorithm.** Policy optimization with no-regret oracle - what does this mean exactly? A no regret policy optimization oracle is an algorithm for which the sequence of policies produced satisfies that the regret is o(K), for K policies or algorithm iterations. A more detailed explanation is under `Algorithm 1`. 

**Deep RL version of algorithm** Algorithm 2 shows a practical implementation of the theory proposed in the paper. The key differences to the classical way of doing things is that 2 critic networks and in the way they are updated. Whilst combining the Bellman consistency with the pessimistic term, they optimize this by projected gradient descent. Why projected gradient descent, which projection? Projected descent ensures bounded complexity for the critic and enables stability across different values of $$\beta$$. The projection is on the space of neural networks of L2 bounded weights, i.e. we do a gradient step to minimize the loss and then we project to the closest poin that satisfies the bound. Apparently using L2 regularization doesn't work as well (in comparison, using the regularization needs to be tuned, whereas the projection ensures momentarily that a hard constraint is satisfied).

**Double Q residual algorithm loss.** They propose an improvemenet to the standard double Q learning to mitigate the deadly triad problem (approximation, bootstrapping, off-policy updates). They do a convex combination of the Bellman consistency losses between using the minimum of the two critics and the critic itself. This kind of update is done for both critics. `This might be in general a useful idea for actor-critic algorithms`, although it adds another hyperparameter to tune that controls the convex combination.

**Policy projection.** They consider the class of policies with a minimal entropy constraint (why wouldn't we want our policies to be arbitrarily confident based on the data? This also means sometimes being so confident that we have close to zero entropy.). This is again optimized by projected gradient descent.

**Policy loss.** Interestingly, the policy optimizes only a single critic, the argument is that taking the minimum of the two critics instead causes instability for small values of $$\beta$$. 

**Experiments.** The main conclusions are that ATAC outperforms the model-free and model-based baselines in the D4RL benchamrk, the DQRA loss (doing the convex combination for the critic losses) seems like affecting the performance strongly. Furthermore, there are experiments showing robustness for different values of $$\beta$$.


**Summary** One of the key questions that is left unanswered in the paper is what is a good choice of the $$\beta$$ hyperparameter?.


## Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

Paper introduces mixture of experts layer as a way of increasing the size of neural networks in terms of number of parameters, but at the same time reducing computation since only subpart of the experts are used for computation on per-sample basis.
The MoE layer consists of simple MLPs and a learnable gating network which selects a sparse combination of experts for computation, so key questions:
1) How do we train this? Does every expert see the same data or separate subsets of data?

All parts are trained jointly by back-propagation (whatever is meant by "jointly", I guess end-to-end). So the key here is that the gating mechanism masks out the output of the individual experts. By the gating mechanism, we can decide which parts of this large layer we want to compute.

**Noisy Top-K Gating** before taking the softmax function, they add tunable Gaussian noise and then keep only the top-k values, the rest being set to -$\infty$. This is a way in which they ensure sparsity, it seems though rather arbitrary, there is no hard-gating, there is only the softmax that is used to compute the weights of each of the outputs. This is literally all that they do, no fancy gradient computation through discrete decisions by REINFORCE. 

**The shrinking batch problem** This is what they call the problem of experts receiving data sparsely, because k < n. To increase the batch size, they propose data parallelism and model parallelism. This seems to me to be ad-hoc. I think the right way of doing things here is to allow the experts to "fight" for data points, this means that inherently some experts might not see some data points and might actually never get selected. This is good, since we are "pruning" our huge model in this way, which leads to less parameters.

**Balancing expert utilization** seems like the same problem as above, some experts get too strong too early and end up being preferred. To combat this, they have an additional loss that enforces that experts have similar importance. Tbh, I don't really understand equation (6), the importance is the sum over gatings for an example x in batch X? Is this the variance across gating dimensions and we want to push the variance to be 0? ... Ah, now I understand, so basically they don't use the variance, but rather the coefficient of variation, which is basically sqrt(variance)/mean. But still, it's not clear to me why would we want to minimize this quantity...


**Summary points.** It seems to me that the actual problem of datapoint assignment hasn't been addressed in this paper, I am also a bit puzzled  by the way how the problem of collapse to a few experts has been solved by gating coefficient of variation, might be that there is a better way of doing this. I like the idea of making a "mixture of experts layer", and then being able to use it as plug and play layer in a neural networks, but stacking these kind of layers together (as far as I know) hasn't been explored. This might also open up a whole different can of worms and might also not be actually useful, need to think about this.


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

**Sample-based action-independent policy improvement** well, basically we sample from a proposal distribution $$\beta$$ finitely many samples and compute the empirical distribution (non-zero only on the samples), this is multiplied by $$f(s,a, Z_\beta(s))$$ where the last term is state-dependent normalizing factor.

At first glance, the theorem in 4.4 seems to a be a straight-forward application of the CLT, to show that with infinite number of samples we converge to the true improvement operator in distribution.


**Section 5, finally, we get to Sampled MuZero** so this part is dealing with how to adapt MCTS, more concretely PUCT formula (Silver et al., 2016) in order to use it in sampled action spaces. One straight-forward approach is search over sampled actions, and keep everything unchanged. This can though lead to unstable results. (it's still unclear to me how the sampling is done here, for each step separately, or we just sample finitely many actions from action space and continue business as usual?). What the authors rather propose is using $$\pi_\beta$$ for action selection, where $$\pi_\beta = (\frac{\hat \beta}{\beta}\pi$$, i.e. an importance weighted policy since the sampling was done from the proposal distribution $$\beta$$. 

Apparently in the theory section the authors argue to use importance sampling where the importance weights are determined by the ratio between the empirical distribution and action sampling distribution - but in the end what they end up doing is just discretisizing the continuous action space apriori, kind of funny.

**this paper needs more attention!**



## Mastering Atari Games with Limited Data - MuZero (Ye et al. 2021), 2022.02.21 

The authors build on top of MuZero Reanalyze algorithm to introduce EfficientZero, with which they achieve SotA performance on the Atari benchmark. They start by noticing 3 things about the MuZero Reanalyze algorithm:
* lack of supervision of the environment model - because the model is only trained through reward, value and policy functions. Reward can be sparse, giving a sparse signal, value functions are noisy because of **bootstrapping**
* problems with aleatoric uncertainty causes predicted rewards to have large prediction errors.
* off-policy issues of multi-step value - MuZero uses multi-step reward from the environment to learn value functions faster, but we run into off-policy evaluation here, which further hinders performance.

**First** improvement is that the outputs of the representation function and the dynamics function should be the same for $$o_{t+1}$, i.e. $s_{t+1} = \hat{s}_{t+1}$$. To enforce this, they use SimSiam (cite), a self-supervised algorithm which essentially pulls projected views of the same thing close together. Note that, $s$ here is an internal state representation of the algorithm.

**Second** they tackle the recurrent error accumulation, which they call the "state aliasing problem". A solution to this is predicting the value prefix, rather than using N-step TD for computing the value function target. The intuition about this is kind of cool - humans don't tend to imagine exactly at which step they will receive a reward, but rather have a rough idea, so the value prefix is just going to be a function of the whole state sequence. Note that, this doesn't require bootstrapping and therefore is less noisy, it's trained via direct supervision. In essence though, this doesn't fix the aleatoric uncertainty issue, maybe a distributional RL approach would be of some use here, or ensemble of value prefixes.

**Third** MuZero Reanalyze reuses the data from the replay buffer, which naturally leads to off-policy issues. They propose to re-use the rewards of a dynamic horizon $l$ and re-computing the rest via MCTS for computing the value target. The horizon is smaller the older the data. This causes increased computation cost, but we don't care about this really, because it can be done in parallel.

**Implementation Technicalities** Interestingly, the internal representation $$s_t$$ that MuZero uses in their case is a very small image, the internal forward model that takes $$(s_t, a_t)$$ is a ResNet, basically everything is a ResNet! Except the value prefix that they compute, kind of weird, why wouldn't we want to learn a low-dimensional vector representation?

**Is the representation good?** The authors do some analysis of the quality of the representation so that they showcase the benefit of using a self-supervised consistency loss. Indeed, if they learn a decoder to decode the observations from the learned representation, something reasonable appears when decoding subsequent predictions of the internal model. For the case of vanilla MuZero, the decoder decodes garbage more-or-less, which is kind of weird. Moreover, `neither this method nor vanilla MuZero showcases in the decoding part that really relevant information for the task is encoded`. For example, in the Atari breakout game, one can't see the ball at all in the decoded frames. This could be though a by-product of the loss function that was used to train the decoder, since the ball is a relatively small part of the image, a simple cross-entropy loss is mostly going to ignore the ball. 

**In summary**, it is interesting to see what kind of improvements are resulting from these  modifications and they do make sense. However, there can be some improvements. Off the top of my head, using an ensemble of stochastic neural networks such as in PETS or RAZER should be able to deal better with aleatoric and epistemic uncertainty from the environment. Furthermore, I am not really convinced by the off-policy correction technique, it seems a bit ad-hoc since it uses this dynamic horizon of re-computation. Ideas from importance sampling could be used here, but they would introduce more variance to the algorithm - could be interesting to explore, nevertheless.






<script> 
var c = 0;
$(function() {
    var num = $("h2").length;
    $("h2").each(function(index){ $(this).html((num-index).toString() + ". " + $(this).html())})
}
);

</script>