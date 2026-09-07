"""
This file owns the entire question of what fuel properties does each position in the domain have. It answers a ScalarFieldProtocol query by computing fuel load at any position from a spatial model of tree placement.

The complexity lives here, in layers you add one at a time:

Layer 1 — uniform (today)
Every cell has the same fuel load. UniformScalarField(0.5). This is not even in random_tree_placement_field.py yet, just the flat field.

Layer 2 — random Poisson placement
Trees are placed as a Poisson point process with density λ trees/m². Each tree contributes a fuel load footprint (a Gaussian kernel of radius r_m). The field sums contributions from all trees within influence distance of the query position. The seed parameter makes it reproducible.

Layer 3 — clustered placement
Replace the Poisson process with a Thomas cluster process (parent points Poisson, children Gaussian around each parent). Models natural forest patching. One extra parameter: cluster radius.

Layer 4 — species heterogeneity
Each tree is assigned a species from a probability distribution. Each species maps to a different FuelProperties preset. The field now returns not a scalar but a FuelProperties object — at this point it graduates from ScalarFieldProtocol to its own FuelFieldProtocol.

Layer 5 — real forest map
Load a raster of classified vegetation (IGN BD Forêt, Copernicus land cover) and map class IDs to FuelProperties presets. The field becomes VegetationRasterField, same protocol, and the fire engine sees nothing different
"""