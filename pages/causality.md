---
layout: default
title: Causality
permalink: /causality
full_bleed: true
---

<main class="page resource-page">

<header>
  <h1>Causality Resources</h1>
  <p>A curated collection of resources on causal inference and causal discovery that I've found useful — talks, lectures, books, websites and papers.</p>
</header>

<h2><i class="fas fa-globe" aria-hidden="true"></i> Websites</h2>

<ul class="resource-list">
{% for site_item in site.data.websites %}
{% if site_item.category == "causality" %}
  <li class="resource-item">
    <a href="{{site_item.link}}" target="_blank" rel="noopener noreferrer">{{site_item.link}}</a>
    {% if site_item.description %}<span class="resource-item__desc">{{site_item.description}}</span>{% endif %}
  </li>
{% endif %}
{% endfor %}
</ul>

<h2><i class="fas fa-video" aria-hidden="true"></i> Talks</h2>

<ul class="resource-list">
{% for talk in site.data.talks %}
{% if talk.category == "causality" %}
  <li class="resource-item">
    <a href="{{talk.link}}" target="_blank" rel="noopener noreferrer">{{talk.description}}</a>
  </li>
{% endif %}
{% endfor %}
</ul>

<h2><i class="fas fa-chalkboard-teacher" aria-hidden="true"></i> Lectures</h2>

<ul class="resource-list">
{% for lect in site.data.lectures %}
{% if lect.category == "causality" %}
  <li class="resource-item">
    <a href="{{lect.link}}" target="_blank" rel="noopener noreferrer">{{lect.description}}</a>
  </li>
{% endif %}
{% endfor %}
</ul>

<h2><i class="fas fa-book" aria-hidden="true"></i> Free Books</h2>

<ul class="resource-list">
{% for book in site.data.books %}
{% if book.category == "causality" %}
  <li class="resource-item">
    <a href="{{book.link}}" target="_blank" rel="noopener noreferrer">{{book.title}}</a>
    {% if book.description %}<span class="resource-item__desc">{{book.description}}</span>{% endif %}
  </li>
{% endif %}
{% endfor %}
</ul>

</main>
