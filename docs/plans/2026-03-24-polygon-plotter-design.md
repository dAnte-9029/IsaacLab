# Polygon Plotter Design

## Context

The user wants a very small helper script for tail-geometry work. The intended workflow is:

1. open a Python file
2. type polygon vertices in order
3. run the script
4. see a 2D polygon plot immediately

This is not a general CAD pipeline or a geometry-processing library. It is a lightweight visualization aid for manually entered coordinates.

## Goal

Add a single-file plotting utility that draws one closed 2D polygon from a list of points defined directly in the script.

## Recommended Approach

Use one standalone `matplotlib` script under `scripts/tools/`.

Why this approach:

- the user can edit coordinates directly in code without extra config files
- `matplotlib` is already a natural fit for quick geometry inspection
- the tool can stay extremely small and easy to modify

## Interface

The script will define:

- `POINTS`: ordered `(x, y)` vertices
- `TITLE`: optional plot title

The script will also support a few minimal CLI flags:

- `--save <path>` to save a PNG/SVG
- `--title <text>` to override the default title
- `--hide-labels` to suppress point-index labels

## Behavior

The script should:

- validate that at least three points are provided
- close the polygon automatically if the last point is not the first point
- plot vertices in order
- optionally annotate each point with its index
- use equal axis scaling so the geometry is not visually distorted

## Testing

Since this is a small plotting utility, the most valuable pure-Python tests are:

- polygon closure helper closes an open polygon
- polygon closure helper leaves an already closed polygon unchanged
- validation rejects fewer than three distinct vertices

## Non-Goals

- multiple polygons in one figure
- interactive editing
- reading YAML/JSON input
- automatic aerodynamic parameter extraction
