# Cosmic Vine - Luis De Cunto

The file creates a "vine" modeled as a cube sequence.
Each new cube originates from the face of the previous one,
with a random rotation.

This version DOES NOT check for self intersection.
One way of implementing it would be to check the minimum distance
between skeleton segments (1-D lines between centers of the cubes)
and set up a threshold for that distance. It would treat the cubes as
spheres but is a possible approximation.

Self-intersection detection pseudo-code (sphere approximation):

```
for each candidate cube:
    new_segment = (current_center, candidate_center)
    for each previous_segment in skeleton:
        dist = min_distance_between_segments(new_segment, previous_segment)
        if dist < threshold:
            reject candidate  # too close, would overlap
    if not rejected:
        accept candidate
```

Threshold ~ side_length treats each cube as its inscribed sphere
(radius = side_length/2). Two spheres overlap when their centers
are closer than the sum of their radii (side_length).
This allows minor corner/edge overlap but prevents bulk intersection.
