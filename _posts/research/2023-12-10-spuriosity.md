---
layout: post
title:  "Spuriosity Didn't Kill the Classifier: Using Invariant Predictions to Harness Spurious Features"
categories: research
author: "Marin Vlastelica"
authors: Cian Eastwood*, Shashank Singh*, Andrei Liviu Nicolicioiu, Marin Vlastelica, Julius von Kügelgen, Bernhard Schölkopf
venue: NeurIPS 2023
paper: https://arxiv.org/abs/2307.09933
bibkey: eastwood2023spuriosity
subtitle: "Spurious features can be safely harnessed at test time — without labels."
---
Instead of discarding "spurious" features whose relationship with the label changes across domains, we show they can be safely exploited in the test domain without labels: pseudo-labels from stable features provide sufficient guidance when stable and unstable features are conditionally independent given the label. Our Stable Feature Boosting algorithm learns an asymptotically-optimal predictor without test-domain labels.
