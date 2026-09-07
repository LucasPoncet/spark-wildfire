"""Same as UniformVectorField but with a named constructor that makes the physical meaning explicit.

wind speed in m/s and bearing in radians. Exists as a separate file because wind will grow: 
the next version is InterpolatedWindField from a weather model, and the file boundary makes that upgrade contained.
"""