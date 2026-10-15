# Repository Data

This directory contains only small data needed for tests, inspection and the
portable demo. It is not the full training dataset.

Tracked content:

- `demo/`: the tiny deployed checkpoint and its limitation/provenance notes;
- `samples/`: synthetic PPM fixtures, four bounded GZ2 JPEG previews, manifests,
  checksums and the measured demo model card.

The full Galaxy Zoo metadata, source images, prepared 80-image pilot, manifests
and training reports remain outside Git. Their location is machine-specific and
not part of the runtime contract.

To prepare a dataset on another machine, reacquire the licensed/provenanced
source material and use `scripts/prepare_galaxy_zoo_subset.py` with a local output
root. Do not commit downloaded archives, full image collections, generated
caches or machine-specific paths.

The repository-local fixtures are enough to run the focused tests and launch the
public-demo equivalent with `data/demo/gz2_pilot_80.pt`.
