---
title: Causal Model Learning and Modular Computation Line of Work
categories: note
layout: post
---

Rough reading notes on causal model learning and modular computation, connecting independent mechanisms to deep learning architectures.

## Neural Production Systems

GNNs are not always well-suited to extract objects from images, for two main reasons:
* They do not predispose interactions to be sparse, as is mostly the case in physical scenes.
* They do not factorize knowledge about entities.

Drawing inspiration from cognitive science, production systems offer a way to merge propositional logic and deep learning — expressing propositional knowledge is not a natural strength of standard neural networks.

The paper builds upon classical **production systems** that operate in a condition-action way by updating working memory or performing an action on the outside world, using MLPs and attention mechanisms to determine rule-entity binding.

* A production system consists of a set of entities and a set of rules.
* Production rules are **modular** — each rule represents a unit of knowledge.
* Production rules are **sparse**, i.e., dependencies among entities are sparse.
* Uses straight-through Gumbel-softmax for rule and slot selection.
* Unlike GNNs where the graph topology is fixed, the topology induced in NPS is dynamic.

## VIM: Variational Independent Modules for Video Prediction

Learns latent representations of objects and discovers causal mechanisms between these objects in representation space.
* Latent states and transition functions of the latent state have an entity-centric inductive bias.
* The transition functions (called modules) are independently parametrized and shared across entities.
* The model outputs a set of categorical selection variables to determine which mechanisms are applied, sampling modules according to their attention importance weights via Gumbel-softmax.
* A module can only consider a handful of slots as its input argument (in this work they consider only unary modules, leaving $n$-ary modules for future work).

## Feature Attending Recurrent Modules for Generalization in Reinforcement Learning

Cognitive scientists theorize that humans generalize broadly using "schemas" they discover for regularly occurring structures — composable representations over portions of our observations.

This work hypothesizes that a single deep RL architecture can exhibit multiple types of generalization if it learns schema-like representations for regularly occurring structures within its experience:
* **FARM** is an architecture designed to discover task-relevant schemas.
* Analogous to word embeddings — a distributed representation that can be composed in various ways.
* To have the modules coordinate what they attend to, they share information using transformer-style attention (Vaswani et al., 2017).
* One of the claims of the paper is that spatial attention can be detrimental to reinforcement learning; FARM attends to an observation with **feature attention** rather than spatial attention.
* The agent takes in a partial observation and task description.

## Object Files and Schemata: Factorizing Declarative and Procedural Knowledge in Dynamical Systems

Uses Gumbel-based hard selection of appropriate schemata to factorize declarative (object state) and procedural (dynamics) knowledge.

## Recurrent Independent Mechanisms

> A complex generative model, temporal or not, can be thought of as the composition of independent mechanisms or "causal" modules.

This corresponds directly to modular Structural Equation Models.

> One may hypothesize that if a brain is able to solve multiple problems beyond a single i.i.d. task, it may exploit the existence of this kind of structure by learning independent mechanisms that can flexibly be reused, composed and re-purposed.

In other words, since mechanisms are invariant under distribution shifts realized by interventions, they remain reusable across different distributions.

> The central question motivating our work is how a gradient-based deep learning approach can learn a representation of high-level variables which favour learning independent but sparsely interacting recurrent mechanisms in order to benefit from such modularity assumption.

* Each RIM (Recurrent Independent Mechanism) is only activated and updated when the input is relevant to it.
* An attention mechanism selects and activates subsets of independent mechanisms.
* RIMs can be used as a drop-in replacement for a GRU or LSTM layer.
