---
title: Object-Centric Causal Model Learning Line of Work
categories: blog
layout: post
---

## Neural Production Systems

GNNs are not suited to extract objects from images, for two main reasons:
* they do not predispose interactions to be sparse, as it is mostly the case
* they do not factorize knowledge about the entities

Inspiration in cognitive science, production systems.
Expressing propositional knowledge is not the strength of current neural networks systems.
Merge propositional logic and deep learning.

The paper builds upon the classical idea of **production systems** that operate in a condition-action way by updating the `working memory` or performing a certain action on the outside world.
MLP and attention mechanism to determine rule-entity binding.

Production system consits of a set of entities and set of rules.
Production rules are `modular`, each production rule represents a unit of knowledge.  
Production rules are `sparse`, i.e. dependencies among entities are sparse.

Use straight-through Gumbel-softmax for rule and slot selection.

The topology of the graph induced in NPS is dynamic, while the topology in GNNs is fixed.

Use Ka physics environment with boxes?


## VIM: Variational Independent Modules for Video Prediction

Learns latent representations of objects and discovers causal mechanisms between these objects (in representations space).
Latent states and transition functions of the latent state have entity-centric inductive bias. 
The transition functions (called modules) are independently parametrized and shared across entities.
The model outputs a set of categorical selection variables to see which mechanisms are applied.
Modules are sampled according to their attention importance weights.

A module can only consider a handful of slots as its input argument.
In this work they consider only unary modules, n-ary they leave for future work. 

Sampling of mechanism to apply via Gumbel-softmax.

## Feature Attending Recurrent Modules for Generalization in Reinforcement Learning

Cognitive scientists theorize that humans generalize broadly with “schemas” they discover for regularly occuring structures.

schemas are composable representations over portions of our observations.

In this work, we hypothesize that we can develop a single deep RL architecture that can exhibit multiple types of generalization if it can
learn schema-like representations for regularly occurring structures within its experience.

FARM - architecutre to discover task-relevant schemas.

Parallel to word embeddings - better representation for words since it can be used in various in various ways (vector addition) in comparison to standard words.

To have the modules coordinate what they attend to, they share information using transformer-style attention (Vaswani et al., 2017).

One of the claims of the paper is that spatial attention can be detrimental to reinforcement learning.

FARM attends to an observation with feature attention as opposed to spatial attention. 

Agent takes in partial observation and task description. 


## Object Files and Schemata: Factorizing Declarative and Procedural Knowledge in Dynamical Systems

Use Gumbel-based hard selection of appropriate schemata.


## Recurrent Independent Mechanisms

"complex generative model, temporal or not, can be thought of as the composition of
independent mechanisms or “causal” modules" - basically we are thinking of purely Structural Equation Models that are modular here.

"ne may hypothesize
that if a brain is able to solve multiple problems beyond a single i.i.d. (independent and identically
distributed) task, they may exploit the existence of this kind of structure by learning independent
mechanisms that can flexibly be reused, composed and re-purposed" - trying to write a translation of this statement here, since mechanisms are not affected by distribution shifts which are realized by interventions, they are reusable across different distributions.



