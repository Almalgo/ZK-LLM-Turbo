"""Shared runtime constants for the current CKKS configuration."""

POLY_MODULUS_DEGREE = 16384
COEFF_MOD_BIT_SIZES = [60, 40, 40, 40, 40, 40, 60]
GLOBAL_SCALE = 2**40
SLOT_COUNT = POLY_MODULUS_DEGREE // 2
