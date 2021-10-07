---
layout: default
title: Machine Learning
permalink: /ml
---


## Free Books

{% for book in site.data.books%}
{% if book.category == "ml"%}

* [{{book.title}}]({{book.link}}) - {{book.description}}
{% endif l%}

{% endfor %}