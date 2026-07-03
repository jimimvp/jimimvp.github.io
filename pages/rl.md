---
layout: default
title: Reinforcement Learning
permalink: /rl
full_bleed: true
---

<main class="page resource-page">

<header>
  <h1>Reinforcement Learning</h1>
  <p>Resources on sequential decision making — reinforcement learning, model-predictive control, and related topics.</p>
</header>

<h2>{% include icon.html name="lecture" %} Lectures</h2>

<ul class="resource-list">
{% for lect in site.data.lectures %}
{% if lect.category == "rl" %}
  <li class="resource-item">
    <a href="{{lect.link}}" target="_blank" rel="noopener noreferrer">{{lect.description}}</a>
  </li>
{% endif %}
{% endfor %}
</ul>

<h2>{% include icon.html name="book" %} Free Books</h2>

<ul class="resource-list">
{% for book in site.data.books %}
{% if book.category == "rl" %}
  <li class="resource-item">
    <a href="{{book.link}}" target="_blank" rel="noopener noreferrer">{{book.title}}</a>
    {% if book.description %}<span class="resource-item__desc">{{book.description}}</span>{% endif %}
  </li>
{% endif %}
{% endfor %}
</ul>

</main>
