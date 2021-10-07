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
{% if talk.category == causality%}

* [{{talk.link}}]({{talk.link}}) - {{talk.description}}

{% endif %}

{% endfor %}



## Lectures

{% for lect in site.data.lectures%}
{% if lect.category == causality%}
* [{{lect.link}}]({{lect.link}}) - {{lect.description}}
{% endif %}

{% endfor %}


## Free Books

{% for book in site.data.books%}
{% if book.category == causality%}
* [{{book.title}}]({{book.link}}) - {{book.description}}
{% endif %}
{% endfor %}