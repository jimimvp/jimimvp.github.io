---
layout: default
title: Causality
permalink: /causality
---

# Causality Resources

Here I have collected various resources on causality that I found useful. This includes talks, books, lecture, websites series and occasionally **papers**

## Websites

* https://causalinference.gitlab.io 


## Talks

{% for talk in site.data.talks%}
* [{{talk.link}}]({{talk.link}}) - {{talk.description}}
{% endfor %}



## Lectures

{% for lect in site.data.lectures%}
* [{{lect.link}}]({{lect.link}}) - {{lect.description}}
{% endfor %}


## Free Books

{% for book in site.data.books%}
* [{{book.title}}]({{book.link}}) - {{book.description}}
{% endfor %}