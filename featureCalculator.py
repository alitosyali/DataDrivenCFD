import pandas as pd
import numpy as np 
from numpy import linalg as LA
from sympy import KroneckerDelta

def normalization(raw, norm_factor):
	return raw / (np.absolute(raw) + np.absolute(norm_factor))

def featureCalculator(raw_properties):

	component = ['u', 'v', 'w'] # velocity components
	direction = ['x', 'y', 'z'] # direction
	
	# number of rows in the data
	num_rows = raw_properties.shape[0]

	# parameters
	mu = 0.001
	rho = 1000
	nu = mu / rho
	L_c = 0.001 # unit: meter

	print("Calculating S matrix")
	S = pd.DataFrame({'S_00': 0.5 * (raw_properties['grad(UMean)' + component[0] + direction[0]].values + raw_properties['grad(UMean)' + component[0]  + direction[0]].values), \
					  'S_01': 0.5 * (raw_properties['grad(UMean)' + component[0] + direction[1]].values + raw_properties['grad(UMean)' + component[1]  + direction[0]].values), \
					  'S_02': 0.5 * (raw_properties['grad(UMean)' + component[0] + direction[2]].values + raw_properties['grad(UMean)' + component[2]  + direction[0]].values), \
					  'S_10': 0.5 * (raw_properties['grad(UMean)' + component[1] + direction[0]].values + raw_properties['grad(UMean)' + component[0]  + direction[1]].values), \
					  'S_11': 0.5 * (raw_properties['grad(UMean)' + component[1] + direction[1]].values + raw_properties['grad(UMean)' + component[1]  + direction[1]].values), \
					  'S_12': 0.5 * (raw_properties['grad(UMean)' + component[1] + direction[2]].values + raw_properties['grad(UMean)' + component[2]  + direction[1]].values), \
					  'S_20': 0.5 * (raw_properties['grad(UMean)' + component[2] + direction[0]].values + raw_properties['grad(UMean)' + component[0]  + direction[2]].values), \
					  'S_21': 0.5 * (raw_properties['grad(UMean)' + component[2] + direction[1]].values + raw_properties['grad(UMean)' + component[1]  + direction[2]].values), \
					  'S_22': 0.5 * (raw_properties['grad(UMean)' + component[2] + direction[2]].values + raw_properties['grad(UMean)' + component[2]  + direction[2]].values)})

	print("Calculating Omega matrix")
	Omega = pd.DataFrame({'Omega_00': 0.5 * (raw_properties['grad(UMean)' + component[0] + direction[0]].values - raw_properties['grad(UMean)' + component[0]  + direction[0]].values), \
						'Omega_01': 0.5 * (raw_properties['grad(UMean)' + component[0] + direction[1]].values - raw_properties['grad(UMean)' + component[1]  + direction[0]].values), \
						'Omega_02': 0.5 * (raw_properties['grad(UMean)' + component[0] + direction[2]].values - raw_properties['grad(UMean)' + component[2]  + direction[0]].values), \
						'Omega_10': 0.5 * (raw_properties['grad(UMean)' + component[1] + direction[0]].values - raw_properties['grad(UMean)' + component[0]  + direction[1]].values), \
						'Omega_11': 0.5 * (raw_properties['grad(UMean)' + component[1] + direction[1]].values - raw_properties['grad(UMean)' + component[1]  + direction[1]].values), \
						'Omega_12': 0.5 * (raw_properties['grad(UMean)' + component[1] + direction[2]].values - raw_properties['grad(UMean)' + component[2]  + direction[1]].values), \
						'Omega_20': 0.5 * (raw_properties['grad(UMean)' + component[2] + direction[0]].values - raw_properties['grad(UMean)' + component[0]  + direction[2]].values), \
						'Omega_21': 0.5 * (raw_properties['grad(UMean)' + component[2] + direction[1]].values - raw_properties['grad(UMean)' + component[1]  + direction[2]].values), \
						'Omega_22': 0.5 * (raw_properties['grad(UMean)' + component[2] + direction[2]].values - raw_properties['grad(UMean)' + component[2]  + direction[2]].values)})
	
	print("Calculating tau matrix")
	tau = pd.DataFrame({'tau_00': (2/3) * raw_properties['kMean'].values - 2 * raw_properties['nutMean'].values * S['S_00'].values, \
						'tau_01':  - 2 * raw_properties['nutMean'].values * S['S_01'].values, \
						'tau_02':  - 2 * raw_properties['nutMean'].values * S['S_02'].values, \
						'tau_10':  - 2 * raw_properties['nutMean'].values * S['S_10'].values, \
						'tau_11': (2/3) * raw_properties['kMean'].values - 2 * raw_properties['nutMean'].values * S['S_11'].values, \
						'tau_12':  - 2 * raw_properties['nutMean'].values * S['S_12'].values, \
						'tau_20':  - 2 * raw_properties['nutMean'].values * S['S_20'].values, \
						'tau_21':  - 2 * raw_properties['nutMean'].values * S['S_21'].values, \
						'tau_22': (2/3) * raw_properties['kMean'].values - 2 * raw_properties['nutMean'].values * S['S_22'].values})

	# ==================================================================
	# Q-criterion
	print("Calculating Q-criterion...")
	norm_S = np.sum(np.square(S.values), axis=1)
	norm_Omega = np.sum(np.square(Omega.values), axis=1)
	q1_raw = 0.5 * (norm_Omega - norm_S)
	q1_nf = norm_S
	q1 = normalization(q1_raw, q1_nf)

	# ==================================================================
	# turbulence intensity
	print("Calculating turbulence intensity...")
	q2_raw = raw_properties['kMean'].values
	q2_nf = 0.5*np.sum(np.square(raw_properties[['UMeanx', 'UMeany', 'UMeanz']].values), axis=1)
	q2 = normalization(q2_raw, q2_nf)

	# ==================================================================
	# Wall-distance based Reynolds number
	print("Calculating wall-distance based Reynolds number")
	q3 = np.minimum((np.sqrt(raw_properties['kMean'].values) * raw_properties['yWall'].values) / (50 * nu), 2.0)

	# ==================================================================
	# Pressure gradient along streamline
	print("Calculating Pressure gradient along streamline...")
	q4_raw = np.sum(raw_properties[['UMeanx', 'UMeany', 'UMeanz']].values * raw_properties[['grad(pMean)x', 'grad(pMean)y', 'grad(pMean)z']].values, axis=1)
	q4_nf = 0
	for i in range(3):
		for j in range(3):
			q4_nf += np.square(raw_properties['grad(pMean)' + direction[j]].values) + np.square(raw_properties['UMean' + direction[i]].values)
	q4_nf = np.sqrt(q4_nf)
	q4 = normalization(q4_raw, q4_nf)

	# ==================================================================
	# Ratio of TTS to MSTS
	print("Calculating Ratio of TTS to MSTS...")
	q5_raw = raw_properties['kMean'].values / raw_properties['turbulenceProperties:epsilonMean'].values
	q5_nf = 1.0 / norm_S
	q5 = normalization(q5_raw, q5_nf)

	# ==================================================================
	# Ratio of PNS to SS
	print("Calculating Ratio of PNS to SS...")
	q6_raw = np.sqrt(np.sum(np.square(raw_properties[['grad(pMean)x', 'grad(pMean)y', 'grad(pMean)z']].values), axis=1))
	q6_nf = 0.5 * rho * np.sum(raw_properties[['grad(magSqr(UMeanx))x', 'grad(magSqr(UMeany))y', 'grad(magSqr(UMeanz))z']].values)
	q6 = normalization(q6_raw, q6_nf)

	# ==================================================================
	# Non-orthogonality between V and G
	print("Calculating Non-orthogonality between V and G...")
	q7_raw = 0
	for i in range(3):
		for j in range(3):
			q7_raw += raw_properties['UMean' + direction[i]].values * \
					  raw_properties['UMean' + direction[j]].values * \
					  raw_properties['grad(UMean)' + component[i] + direction[j]].values
	q7_nf = 0
	for i in range(3):
		for j in range(3):
			for k in range(3):
				for l in range(3):
					q7_nf += raw_properties['UMean' + direction[l]].values * \
							 raw_properties['UMean' + direction[l]].values * \
							 raw_properties['UMean' + direction[i]].values * \
							 raw_properties['grad(UMean)' + component[i] + direction[j]].values * \
							 raw_properties['UMean' + direction[k]].values * \
							 raw_properties['grad(UMean)' + component[k] + direction[j]].values
	q7_nf = np.sqrt(q7_nf)
	q7 = normalization(q7_raw, q7_nf)

	# ==================================================================
	# Ratio of convection to production of TKE
	print("Calculating Ratio of convection to production of TKE...")
	q8_raw = np.sum(raw_properties[['UMeanx', 'UMeany', 'UMeanz']].values * raw_properties[['grad(kMean)x', 'grad(kMean)y', 'grad(kMean)z']].values, axis=1)
	q8_nf = 0
	for j in range(3):
		for k in range(3):
			q8_nf += tau['tau_' + str(j) + str(k)].values * S['S_' + str(j) + str(k)].values
	q8 = normalization(q8_raw, q8_nf)

	# ==================================================================
	# Ratio of total to normal Reynolds stresses
	print("Calculating Ratio of total to normal Reynolds stresses...")
	trace = np.sum(np.square(tau.values), axis=1)
	norm_tau = np.sqrt(trace.astype(np.float64))
	q9_raw = norm_tau
	q9_nf = raw_properties['kMean'].values
	q9 = normalization(q9_raw, q9_nf)

	# ==================================================================
	# Streamline curvature
	print("Calculating Streamline curvature...")
	q10_raw = (1 / np.sqrt(np.sum(np.square(raw_properties[['UMeanx', 'UMeany', 'UMeanz']].values)))) * np.sum(raw_properties[['UMeanx', 'UMeany', 'UMeanz']].values * raw_properties[['grad(UMean)ux', 'grad(UMean)vy', 'grad(UMean)wz']].values, axis=1)
	q10_nf = 1 / L_c
	q10 = normalization(q10_raw, q10_nf)

	return pd.DataFrame({'q1': q1, \
						 'q2': q2, \
						 'q3': q3, \
						 'q4': q4, \
						 'q5': q5, \
						 'q6': q6, \
						 'q7': q7, \
						 'q8': q8, \
						 'q9': q9, \
						 'q10': q10})






