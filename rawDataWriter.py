import pandas as pd 
import os
import numpy as np 
import glob

from rawDataReader import rawDataReader

def rawDataWriter(predictions, discrapency_name, case, take_log):

	# parent directory
	parent_directory = os.path.dirname(os.getcwd())

	preds = predictions
	files = glob.glob(parent_directory + '\data\RANS\\' + case + '\\' + discrapency_name)
	# files = glob.glob(parent_directory + '/data/RANS/' + case + '/' + discrapency_name)
	rans = rawDataReader(files)

	num_obs = rans.shape[0]

	preds = preds['predictions'].values
	if take_log:
		rans = np.log10(rans[discrapency_name].values)
	else:
		rans = rans[discrapency_name].values

	with open(parent_directory + '\\results\\' + discrapency_name + 'Pred' + case + '.txt', 'w') as f:
	# with open(parent_directory + '/results/' + discrapency_name + 'Pred' + case + '.txt', 'w') as f:
		f.write('/*--------------------------------*- C++ -*----------------------------------*\\')
		f.write('\n| =========                 |                                                 |')
		f.write('\n| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |')
		f.write('\n|  \\\\    /   O peration     | Version:  v1706                                 |')
		f.write('\n|   \\\\  /    A nd           | Web:      www.OpenFOAM.com                      |')
		f.write('\n|    \\\\/     M anipulation  |                                                 |')
		f.write('\n\*---------------------------------------------------------------------------*/')
		f.write('\nFoamFile')
		f.write('\n{')
		f.write('\n    version     2.0;')
		f.write('\n    format      ascii;')
		f.write('\n    class       volScalarField;')
		f.write('\n    location    "5000";')
		f.write('\n    object      {};'.format(discrapency_name + 'Pred'))
		f.write('\n}')
		f.write('\n// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //')
		f.write('\n')
		f.write('\ndimensions      [0 2 -2 0 0 0 0];')
		f.write('\n')
		f.write('\n')
		f.write('\ninternalField   nonuniform List<scalar> ')
		f.write('\n2880000')
		f.write('\n(')
		for i in range(num_obs):
			f.write('\n' + str(rans[i] + preds[i]))
		f.write('\n)')
		f.write('\n;')
		f.write('\n')
		f.write('\nboundaryField')
		f.write('\n{')
		f.write('\n    inlet')
		f.write('\n    {')
		f.write('\n        type            fixedValue;')
		f.write('\n        value           uniform 0;')
		f.write('\n    }')
		f.write('\n')
		f.write('\n    outlet')
		f.write('\n    {')
		f.write('\n        type            fixedValue;')
		f.write('\n        value           uniform 0;')
		f.write('\n    }')
		f.write('\n')
		f.write('\n    top')
		f.write('\n    {')
		f.write('\n        type            fixedValue;')
		f.write('\n        value           uniform 0;')
		f.write('\n    }')
		f.write('\n')
		f.write('\n    bottom')
		f.write('\n    {')
		f.write('\n        type            fixedValue;')
		f.write('\n        value           uniform 0;')
		f.write('\n    }')
		f.write('\n')
		f.write('\n    back')
		f.write('\n    {')
		f.write('\n        type            cyclic;')
		f.write('\n    }')
		f.write('\n')
		f.write('\n    front')
		f.write('\n    {')
		f.write('\n        type            cyclic;')
		f.write('\n    }')
		f.write('\n')
		f.write('\n    spacer')
		f.write('\n    {')
		f.write('\n        type            fixedValue;')
		f.write('\n        value           uniform 0;')
		f.write('\n    }')
		f.write('\n}')
		f.write('\n')
		f.write('\n')
		f.write('\n// ************************************************************************* //')


