---
layout: default
title: Reinforcement Learning
permalink: /rl
---

Here are some resources about sequential decision making, this includes reinforcement learning and model-predictive control.


## Lectures

{% for lect in site.data.lectures%}
{% if lect.category == "rl"%}
* [{{lect.link}}]({{lect.link}}) - {{lect.description}}
{% endif %}
{% endfor %}


## Free Books

{% for book in site.data.books%}
{% if book.category == "rl"%}

* [{{book.title}}]({{book.link}}) - {{book.description}}
{% endif l%}

{% endfor %}