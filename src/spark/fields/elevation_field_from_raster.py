"""Loads a DEM file (GeoTIFF, IGN RGE Alti, SRTM) and answers elevation queries.

Wraps a bilinear interpolator. The mesh constructor calls this at build time
to set the z-coordinate of each cell and compute slope angles carried in
neighbor_unit_directions_xyz. After mesh construction this field is no longer
needed — the geometry is baked in.
"""
