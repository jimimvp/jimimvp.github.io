---
title: Generative Flow Networks
layout: post
categories: note
--- 

New work really being pushed by Yoshua Bengio, a model class that enables tractable probabilistic inference and has connections to Markov Chain Monte-Carlo with also the ability to learn densities.

This article is based on a read-through of the following papers:
* [Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation](https://openreview.net/pdf?id=Arn2E4IRjEB)
* [GFlowNet Foundations](https://arxiv.org/pdf/2111.09266.pdf)


## Flow Network based Generative Models for Non-Iterative Diverse Candidate Generation

I'm writing this consecutively while reading. So, the paper's abstract says that the goal is to learn a **stochastic policy** that is able to sample multiple high-reward solutions, not "overfit" to a single one and that the probability of sampling this should be proportional to the reward.
Couple of questions to this that I have off the top of my head:
* In RL, we want to learn an optimal policy, i.e. policy that maximizes return - is there a definition for optimality for the case where there are multiple modes of optimality?
* The idea to tackle this problem is not novel, I wonder where is the connection to distributional RL and generative models such as normalizing flows (which are fully capable of learning multi-modal distributions, and are actually universal approximators of distributions) and conditional flows.
So the paper sells itself as tackling an RL problem, but it's actually a distribution modelling problem. 

> A motivating application for this setup is iterative black-box optimization where the learner has
> access to an oracle which can compute a reward for a large batch of candidates at each round, e.g., in
> drug-discovery applications.

This smells of basically specifying the density that is required to be modelled (i.e. be proportional to the reward.)

They start off with defining the policy, proportional to the rewrad (it's crucial to remember, it's for terminal state)
$$
\pi(x) \approx \frac{R(x)}{Z} 
$$
where $$Z$$ is the normalizing constant to make it a distribution, why is here an approximate sign and not an equals sign? Because in the end we want to focus on the max rewards, i.e. decrease the temperature somewhat?

In Section 2. on the end they describe the actual problem of mapping an action sequence to a terminal state, it is straight forward when we have unique assignment, i.e. when the functional mapping is bijective, but problems arise when it's not. The resulting structure that describes this mapping is a DAG (directed acyclic graph). 

I am skipping **Prop 1.** for now, there is some weirdness happening there with action sequence, `come back later`. The gist of it is that standard approaches to modelling the policy with probability distributions are heavily biased for longer action sequences. 


**Seeing MDP as a Flow network.** We can leverage the DAG structure of the MDP and learn a flow $F$, where we write $F(s,a)$ for flow going from $s \mapsto s'$ and $F(s)$ for total flow in $s$. We look at the initial state $s_0$ as the root node (source) that has an in-flow $Z$  and one sink for each leaf node (sink) with out-flow $R(x)$. The flow conditions need to be satisfied, meaning that the incoming flow in a node needs to be equal the outgoing flow (no node-state can generate additional flow). This is simplified when we assume 0 reward for internal nodes and only receiving reward at the terminal states (sinks). There is a close connection of these flow consistency equations and the standard value function in RL, there is also a recursive formulation, the difference seems to be that the expectation of the flow is not taken? The flow consistency can be simply written as

$$ \sum_{s,a: T(s,a)=s'} F(s,a) = R(s') + \sum_{a' \in \mathcal{A}(s')} F(s', a')
$$


**Proposition 2., i.e. flow formulation makes sense and we obtain the densities.** This is a nice little piece of theory. Let $\pi(s)$ denote the probability of landing in state $s$ when starting from $s_0$ and let $x$ denote a terminal state. It shows that if we define the policy as

$$ \pi(a \vert s) = \frac{F(s,a)}{F(s)},
$$

which basically says that we prioritize actions that have higher flow, remember that $F(s,a) = F(s)$. Furthermore, we obtain other interesting results that are useful, the probability of visiting a state is

$$ \pi(s) = \frac{F(s)}{F(s_0)},
$$

where the flow in the initial state $s_0 = \sum_{x \in \mathcal{X}} R(x)$, i.e. sum of all terminal rewards.
Moreover, the probability of ending in the terminal state $x$ is

$$ \pi(x) = \frac{R(x)}{\sum_{x' \in \mathcal{X} R(x')}}
$$
`todo: check proof`.


**How do we train this stuff?** Since the flow consistency satisfies a recursive relations, we can use ideas from deep RL, i.e. TD-learning in order to train this by doing something like an approximate Bellman backup. The argument presented in the paper why we shouldn't use the naive TD-error learning rule is that the flow in the beginning, closer to the root node, can be very large because of the summations (this is because there is no convex combination of the rewards, i.e. the expectation is not calculated).  This is especially apparent in large state spaces where we have a big variety of trajectories (lots of paths to different terminal states). And, voilla, they decide to do a practicality that you would always do with large stuff - take the logarithm - which leads to a natural interpretation that it enforces flow ratio = 1. But I am still unsure about the learning dynamics of this... Moreover, the way they talk about the $\epsilon$ that enforces that the term in the logarithm is not $0$ is somewhat weird, the interpretation is that $\epsilon$ indicates how much we care about smaller flows vs. big ones, it's something like a temperature. 

**Proposition 3.** Shows that training can be off-policy, so the data doesn't need to come from the optimal policy, it's kind of unsurprising, but might be worth looking into the proof nevertheless.

**Experiments.** Just flew over them, they have interesting results regarding state visitation (exploration) in comparison to PPO and MCMC and also molecule generation.


**Summary so far.** It seems like an interesting approach to defining a novel class of policies and enables us to compute much more than action probabilities for the policy following the reward-proportional distribution. It is limited though in the current formulation to the deterministic setting, so some work might be needed there. Also, continuous action spaces are not addressed here, in the case of continuous action spaces the flow consistency sums would turn into integrals, hm...