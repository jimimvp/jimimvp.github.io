---
title: MuZero Line of Work (wip)
categories: note
layout: post
---

It all began with AlphaGo. Where are we now? MuZero. 

## AlphaGo

Mastering the game of go with expert imitation.

## AlphaGo Zero

Mastering go from "scratch" with self-play, but with a game simulator still.

## AlphaStar

Mastering Starcraft II.

##  Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model

Introducing MuZero, still relies heavily on MCTS, but the key point is that the model doesn't learn to simulate game dynamics, but rather has its own latent dynamics and focuses on extracting a value function, policy and reward function.

## Muesli: Combining Improvements in Policy Optimization

Combination of various improvements.

## Mastering Atari Games with Limited Data (NeurIPS 2021)

Improvements to MuZero to make it more sample efficient.
https://github.com/YeWR/EfficientZero


## Online and Offline Reinforcement Learning by Planning with a Learned Model (NeurIPS 2021)

Improvements to MuZero.

# Continuous State Spaces?

The problem of MuZero and AlphaZero is that the action space is configured to be discrete.
The following works expand upon the MCTS idea and introduce continuous action spaces.

## A0C: Alpha Zero in Continuous Action Space

AlphaZero style algorithm with continuous action spaces.

## Learning and Planning in Complex Action Spaces (ICML 2021)

MuZero in arbitrary action spaces.


