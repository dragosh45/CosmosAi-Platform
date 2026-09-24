# GZ2 Pilot Preview Samples

This folder contains a tiny repository-local preview copied from the full
bounded pilot stored on Kingston. It is intentionally not the full dataset.

## Real JPEG preview

galaxy_gz2_pilot_preview_manifest.csv points to four copied 424 x 424 RGB
JPEGs, one for each current project label:

- elliptical
- spiral
- lenticular
- irregular

The JPEGs are copied examples from:

/media/h1dr0/KINGSTON/cosmosai-data/galaxy/

The source pilot remains the authoritative dataset. These files make it
possible to inspect and test the shared manifest/image-loading code without
mounting Kingston.

## Derived PPM teaching sample

images/processed/images_3x3/gz2-pilot-lenticular-3x3.ppm is a 3 x 3 P3
PPM made from the lenticular preview JPEG using a BOX downsample. It is
similar in shape to the original hand-readable fixtures, but it is not an
original GZ2 file and should not be treated as a new scientific image.

Its one-row manifest is galaxy_gz2_pilot_ppm_manifest.csv.

## Inspect with the existing loader

    .venv/bin/python scripts/load_galaxy_manifest.py       data/samples/galaxy_gz2_pilot_preview_manifest.csv       --galaxy-data-root data/samples --check-images

The old artificial fixtures remain in
data/samples/images/processed/images_224/ and are unchanged.
