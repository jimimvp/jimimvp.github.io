---
layout: default
title: Reinforcement Learning
permalink: /rl
---

Here are some resources about sequential decision making, this includes reinforcement learning and model-predictive control.


## Lectures

{% for lect in site.data.lectures%}
* [{{lect.link}}]({{lect.link}}) - {{lect.description}}
{% endfor %}


## Free Books

{% for book in site.data.books%}
* [{{book.title}}]({{book.link}}) - {{book.description}}
{% endfor %}