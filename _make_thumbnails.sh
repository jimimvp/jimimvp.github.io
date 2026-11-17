#mogrify -path tn -thumbnail 160x160 images/*.png
mogrify -path tn -thumbnail 300x300 images/*.png
mogrify -path tn -thumbnail 400x400 images/profile.jpg
for f in tn/*.png; do cwebp -q 82 "$f" -o "${f%.png}.webp" -quiet; done
cwebp -q 85 images/profile.jpg -o images/profile.webp -quiet
