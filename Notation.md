# Notation

Mapping between the symbols used in the source papers and the identifiers used in this repository.

Paper notation is permitted **only inside the body of a function whose name states which published equation it implements**. Every signature, dataclass field, return value and module-level constant uses the descriptive name.

---

## Fuel properties

Balbi et al. (2009, 2020), Table 1.

| Symbol | Code name | Unit | Fixed value |
|---|---|---|---|
| `s` | `surface_area_to_volume_ratio_per_m` | m⁻¹ | — |
| `h` | `fuel_bed_depth_m` | m | — |
| `β` | `packing_ratio` | — | — |
| `σ` | `fuel_load_kg_per_m2` | kg m⁻² | — |
| `σ_u` | `effective_fuel_load_kg_per_m2` | kg m⁻² | — |
| `σ_w` | `fuel_water_load_kg_per_m2` | kg m⁻² | — |
| `ρ_v` | `fuel_density_kg_per_m3` | kg m⁻³ | — |
| `m` | `moisture_content_fraction` | — | — |
| `c` | `char_fraction` | — | — |
| `S` | `leaf_area_per_square_metre` | m² m⁻² | — |
| `C_p` | `fuel_specific_heat_j_per_kg_k` | J kg⁻¹ K⁻¹ | — |

---

## Fire geometry and dynamics

| Symbol | Code name | Unit | Fixed value |
|---|---|---|---|
| `R` | `rate_of_spread_m_per_s` | m s⁻¹ | — |
| `R_b` | `rate_of_spread_flame_base_radiation_m_per_s` | m s⁻¹ | — |
| `R_c` | `rate_of_spread_convection_m_per_s` | m s⁻¹ | — |
| `R_r` | `rate_of_spread_flame_radiation_m_per_s` | m s⁻¹ | — |
| `γ` | `flame_tilt_angle_rad` | rad | — |
| `γ_c` | `convective_flow_angle_rad` | rad | — |
| `α` | `slope_angle_rad` | rad | — |
| `H` | `flame_height_m` | m | — |
| `l` | `flame_length_m` | m | — |
| `L` | `flame_depth_m` | m | — |
| `δ` | `extinction_depth_m` | m | — |
| `τ` | `flame_residence_time_s` | s | — |
| `T` | `mean_flame_temperature_k` | K | — |
| `q` | `ignition_energy_j_per_kg` | J kg⁻¹ | — |
| `A` | `radiative_coefficient` | — | — |

---

## Flow

| Symbol | Code name | Unit | Fixed value |
|---|---|---|---|
| `U` | `wind_speed_along_front_normal_m_per_s` | m s⁻¹ | — |
| `U(L)` | `wind_speed_at_flame_base_interface_m_per_s` | m s⁻¹ | — |
| `u_0` | `upward_gas_velocity_flat_terrain_m_per_s` | m s⁻¹ | — |
| `u_c` | `upward_gas_velocity_flame_base_top_m_per_s` | m s⁻¹ | — |

---

## Universal constants

Fixed across all experiments. Defined once as module-level constants.

| Symbol | Code name | Unit | Value |
|---|---|---|---|
| `K_1` | `DRAG_COEFFICIENT_S_PER_M` | s m⁻¹ | `130.0` |
| `s_t` | `AIR_TO_PYROLYSIS_GAS_MASS_RATIO` | — | `17.0` |
| `r_00` | `RADIATIVE_FRACTION_MODEL_COEFFICIENT` | — | `2.5e-5` |
| `τ_0` | `FLAME_RESIDENCE_TIME_PARAMETER_S_PER_M` | s m⁻¹ | `75591.0` |
| `B` | `STEFAN_BOLTZMANN_CONSTANT_W_PER_M2_K4` | W m⁻² K⁻⁴ | `5.6e-8` |
| `ΔH` | `HEAT_OF_COMBUSTION_PYROLYSIS_GASES_J_PER_KG` | J kg⁻¹ | `1.74e7` |
| `Δh` | `HEAT_OF_LATENT_EVAPORATION_J_PER_KG` | J kg⁻¹ | `2.3e6` |
| `χ` | `RADIATIVE_FRACTION` | — | `0.2` |
| `χ_0` | `RADIANCE_COEFFICIENT` | — | `0.3` |
| `C_pw` | `WATER_SPECIFIC_HEAT_J_PER_KG_K` | J kg⁻¹ K⁻¹ | `4180.0` |
| `C_pa` | `AIR_SPECIFIC_HEAT_J_PER_KG_K` | J kg⁻¹ K⁻¹ | `1150.0` |
| `T_a` | `AMBIENT_TEMPERATURE_K` | K | `300.0` |
| `T_i` | `IGNITION_TEMPERATURE_K` | K | `600.0` |
| `T_vap` | `VAPORISATION_TEMPERATURE_K` | K | `373.0` |
| `g` | `GRAVITATIONAL_ACCELERATION_M_PER_S2` | m s⁻² | `9.81` |

---

## Equation index

Where each published equation is implemented.

| Equation | Source | Implemented in |
|---|---|---|
| Flame tilt angle | Balbi 2009 Eq. 2 | `compute_flame_tilt_angle_rad` |
| Rate of spread, closed form | Balbi 2009 Eq. 11a–11b | `compute_rate_of_spread_balbi_2009` |
| Rate of spread, fixed point | Balbi 2020 Eq. 28 | `compute_rate_of_spread_balbi_2020` |
| Ignition energy | Balbi 2020 Eq. 9 | `compute_ignition_energy_j_per_kg` |
| Extinction depth | Balbi 2020 Eq. 12 | `compute_extinction_depth_m` |
| Flame base radiation contribution | Balbi 2020 Eq. 13 | `compute_rate_of_spread_flame_base_radiation` |
| Radiative coefficient | Balbi 2020 Eq. 16 | `compute_radiative_coefficient` |
| Flame height | Balbi 2020 Eq. 23 | `compute_flame_height_m` |
| Convective contribution | Balbi 2020 Eq. 27 | `compute_rate_of_spread_convection` |
| Upward gas velocity | Balbi 2020 Eq. B9 | `compute_upward_gas_velocity` |
| Mean flame temperature | Balbi 2020 Eq. B11 | `compute_mean_flame_temperature_k` |

---

## Acoustic

| Symbol | Code name | Unit |
|---|---|---|
| `s(t)` | `source_signal` | — |
| `h(t)` | `channel_impulse_response` | — |
| `α_att` | `attenuation_coefficient_per_m` | m⁻¹ |
| `r` | `source_receiver_distance_m` | m |
| `θ` | `front_bearing_rad` | rad |
| — | `gain_matrix_source_by_receiver` | — |
| — | `receiver_signal_levels` | — |

---

## Balbi 2009 — implemented model

The constants above are the Balbi **2020** set. Balbi 2009, implemented in
`rate_of_spread_equations.py`, uses its own values. Only the constants the
code actually uses are defined; the rest arrive with Eq. 3 and Eq. 4.

| Symbol | Code name | Unit | Value |
|---|---|---|---|
| `T_a` | `AMBIENT_TEMPERATURE_K` | K | `300.0` |
| `T_i` | `IGNITION_TEMPERATURE_K` | K | `600.0` |
| `Δh_v` | `LATENT_HEAT_OF_EVAPORATION_J_PER_KG` | J kg⁻¹ | `2.3e6` |
| `C_pv` | `FUEL_SPECIFIC_HEAT_J_PER_KG_K` | J kg⁻¹ K⁻¹ | `2000.0` |
| `a` | `MOISTURE_WEIGHTING_COEFFICIENT` | — | `0.05` |
| `R_00` | `BASE_MASS_FLUX_KG_PER_M2_S` | kg m⁻² s⁻¹ | `0.05` |
| `u_00` | `BASE_VELOCITY_COEFFICIENT_M3_PER_KG` | m³ kg⁻¹ | `80.0` |
| — | `RADIANT_FRACTION_ROS_MULTIPLIER` | — | `12.0` |
| `A_0` | `DRY_RADIATIVE_COEFFICIENT` | — | `2.25` |
| — | `MAXIMUM_FLAME_TILT_ANGLE_RAD` | rad | `1.4` |

### Fuel bed

| Symbol | Code name | Unit |
|---|---|---|
| `ρ_v` | `fuel_density_kg_per_m3` | kg m⁻³ |
| `m` | `moisture_content_fraction` | — |
| `s_v` | `surface_area_to_volume_ratio_per_m` | m⁻¹ |
| `σ` | `fuel_load_kg_per_m2` | kg m⁻² |
| `τ` | `residence_time_s` | s |
| `e` | `fuel_bed_depth_m` | m |

### Derived, as pure functions

| Symbol | Function | Unit |
|---|---|---|
| `β` | `compute_packing_ratio` | — |
| `δ` | `compute_optical_depth_m` | m |
| `μ` | `compute_absorption_coefficient` | — |
| `u_0` | `compute_upward_gas_velocity_m_per_s` | m s⁻¹ |
| `q` | `compute_ignition_energy_j_per_kg` | J kg⁻¹ |
| `1 + a m` | `compute_moisture_damping_factor` | — |
| `R_0` | `compute_base_rate_of_spread_m_per_s` | m s⁻¹ |
| `A` | `compute_radiative_coefficient` | — |
| `ν_0` | `compute_radiant_fraction_velocity_m_per_s` | m s⁻¹ |
| `γ` | `compute_flame_tilt_angle_rad` | rad |
| `R` | `compute_rate_of_spread_balbi_2009` | m s⁻¹ |
| `r` | `compute_reduced_rate_of_spread_balbi_2009` | — |

**`m` is a fraction in `FuelProperties` and a percentage in Eq. 14 and 15.**
`compute_moisture_damping_factor` is the single place that converts, so no
other function has to remember.

