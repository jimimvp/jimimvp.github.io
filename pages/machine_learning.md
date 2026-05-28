---
layout: default
title: Machine Learning
permalink: /ml
full_bleed: true
---

<main class="page resource-page">

<header>
  <h1>Machine Learning</h1>
  <p>Books, websites and reference material for machine learning practitioners.</p>
</header>

<h2><i class="fas fa-book" aria-hidden="true"></i> Free Books</h2>

<ul class="resource-list">
{% for book in site.data.books %}
{% if book.category == "ml" %}
  <li class="resource-item">
    <a href="{{book.link}}" target="_blank" rel="noopener noreferrer">{{book.title}}</a>
    {% if book.description %}<span class="resource-item__desc">{{book.description}}</span>{% endif %}
  </li>
{% endif %}
{% endfor %}
</ul>

<h2><i class="fas fa-globe" aria-hidden="true"></i> Websites</h2>

<ul class="resource-list">
{% for site_item in site.data.websites %}
{% if site_item.category == "ml" %}
  <li class="resource-item">
    <a href="{{site_item.link}}" target="_blank" rel="noopener noreferrer">{{site_item.link}}</a>
    {% if site_item.description %}<span class="resource-item__desc">{{site_item.description}}</span>{% endif %}
  </li>
{% endif %}
{% endfor %}
</ul>

</main>
