"""Loads a DEM file (GeoTIFF, IGN RGE Alti, SRTM), wraps a bilinear interpolator, and answers elevation queries at arbitrary positions.

The mesh constructor calls this at build time to set the z-coordinate of each cell and
compute slope angles in neighbor_unit_directions_xyz.
After mesh construction this field is no longer needed — the geometry is baked in.
"""