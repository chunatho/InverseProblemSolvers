# ----------------- #
# DECLARE CONSTANTS #
# ----------------- #
hbar_ergs = 1.0545721e-27 # reduced Planks constant in erg s (erg = g cm^2 / s^2)
hbar_eVs = 6.582119569e-16 # reduced plank's constant

aB_cm = 5.2917721e-9 # Bohr radius in cm

me_g = 9.10938e-28 # electron mass in g
mD_g = 2.014*1.66e-24 # deuterium nucleus - g
mT_g = 3.016*1.66e-24 # tritium nucleus - g
mC_g = 12*1.66e-24
e2_statC = (4.80320e-10)**2 # statC^2 = g cm^3 / s^2
e2_eVcm = 1.43996e-7 # fundamental charge ( statC -> eV*cm )
#e2_eVcm = erg_to_eV*(4.8032047e-10)**2 # statC^2 = cm^3 g / s^2

Z_e = -1.0
Z_H = 1.0
Z_He = 2.0
Z_C = 6.0


# ------------------- #
# DECLARE CONVERSIONS #
# ------------------- #
#erg_to_eV = 6.24150907e11 # g cm^2 / s^2 = 6.2415e11 eV
erg_to_eV = 1./1.602177e-12 # unit conversion
erg_to_eV = 1.602177e-12 # unit conversion

#erg_to_Hartree = 1./1.602177e-12/27.211
erg_to_Hartree = 2.2937104486e10
Hartree_to_erg = 1./2.2937104486e10

Hartree_to_eV = 27.2114079527 # unit conversion
eV_to_Hartree = 1./27.2114079527 # unit conversion
