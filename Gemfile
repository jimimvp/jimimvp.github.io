source "https://rubygems.org"

# Matches the GitHub Pages production build environment exactly.
# Usage:
#   bundle install
#   bundle exec jekyll serve --livereload
gem "github-pages", "~> 232", group: :jekyll_plugins

group :jekyll_plugins do
  gem "jekyll-sitemap"
  gem "jekyll-feed"
  gem "jekyll-redirect-from"
end

# Required for `jekyll serve` on Ruby >= 3.0
gem "webrick"

# Stdlib gems removed from default gems in Ruby >= 3.4, still required by jekyll 3.9
gem "csv"
gem "base64"
gem "bigdecimal"
gem "logger"
