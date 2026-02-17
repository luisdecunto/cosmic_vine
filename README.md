# Cosmic Vine

Cosmic Vine generates a 3D chain of connected cubes using local-to-global transformations.
Each new cube is attached to a random face of the previous cube and receives a random local rotation.
The final geometry is exported as a standard OBJ mesh.

## How to Run

```bash
python cosmic_vine.py
```

## Dependencies

- Python 3.10+
- `numpy`

Install dependency:

```bash
python -m pip install numpy
```

## Output

- `cosmic_vine_{n_gen}.obj`: OBJ file containing the generated cube vine mesh with `n_gen` cubes
- Console message with generation summary (example: `Generated 15 cubes -> cosmic_vine_15_cubes.obj`)

## Example

Screenshot of `examples/cosmic_vine_87_cubes.obj`:

![Cosmic Vine Example](examples/cosmic_vine_example.png)

- OBJ file: [examples/cosmic_vine_87_cubes.obj](examples/cosmic_vine_87_cubes.obj)

## Notes

- Self-intersections are intentionally not handled in this version (per assessment guidance).

Self-intersection detection idea (sphere approximation):

```text
for each candidate cube:
    new_segment = (current_center, candidate_center)
    for each previous_segment in skeleton:
        dist = min_distance_between_segments(new_segment, previous_segment)
        if dist < threshold:
            reject candidate
    if not rejected:
        accept candidate
```

Threshold ~ `side_length` treats each cube as its inscribed sphere (`radius = side_length / 2`).
