import numpy as np 
import pandas as pd

def rawDataReader(files):

	'''
	This function takes the names of the files to read and return a
	dataframe which contains all variables of a simulation result.

	INPUT:
		files: a list that contains names of the files 

	OUTPUT:
		a dataframe that contains all variables coming from different 
		raw data file.

	Example:
		df = rawDataReader(files=['U.dat', 'p.dat'])

	Written by Ali Tosyali and Mustafa Usta
	'''

	# an empty list for dataframes
	dfs = []

	for file in files:

		if not 'Readme' in file and not 'Icon' in file:

			vector = False
			tensor = False
			scalar = False

			# name of the variable
			var_name = file[file.find("case_")+8:]#file.find(".")]
			print("Now reading {}".format(var_name))

			# create an empty list for the lines to extract from the file
			data = []

			# check if the file contains vector variable or not.
			if 'scalar' in open(file).read():
				scalar = True
			elif 'vector' in open(file).read():
				vector = True
			else:
				tensor = True

			# obtaining the number of observations from the file
			lookup = 'internalField'
			with open(file) as f:
				for i, line in enumerate(f):
					if lookup in line:
						num_obs_line = i+1

			# if it is vector variable
			if vector:
				# open the file as f
				with open(file) as f:
					# for each line of the file
					for i, line in enumerate(f):
						if i == num_obs_line:
							# obtain the number of observations
							num_obs = int(line)
						if i >= num_obs_line+2 and i <= (num_obs + num_obs_line+1):
							# extract the values between parantheses
							line_vals = line[line.find("(")+1:line.find(")")]
							# convert the values into floating points
							line_vals = [float(s) for s in line_vals.split(' ')]
							# put the values in the data list
							data.append(line_vals)
					# put the all values of this variable into a dataframe
					df = pd.DataFrame(np.array(data), columns = [var_name+"x", var_name+"y", var_name+"z"])

			# if it is a scalar variable
			elif scalar:
				with open(file) as f:
					# for each line of the file
					for i, line in enumerate(f):
						if i == num_obs_line:
							# obtain the number of observations
							num_obs = int(line)
						if i >= num_obs_line+2 and i <= (num_obs + num_obs_line+1):
							data.append(float(line))
					df = pd.DataFrame(np.array(data), columns = [var_name])

			elif tensor:
				# open the file as f
				with open(file) as f:
					# for each line of the file
					for i, line in enumerate(f):
						if i == num_obs_line:
							# obtain the number of observations
							num_obs = int(line)
						if i >= num_obs_line+2 and i <= (num_obs + num_obs_line+1):
							# extract the values between parantheses
							line_vals = line[line.find("(")+1:line.find(")")]
							# convert the values into floating points
							line_vals = [float(s) for s in line_vals.split(' ')]
							# put the values in the data list
							data.append(line_vals)
					# put the all values of this variable into a dataframe
					df = pd.DataFrame(np.array(data), columns = [var_name+"ux", var_name+"uy", var_name+"uz", \
																 var_name+"vx", var_name+"vy", var_name+"vz", \
																 var_name+"wx", var_name+"wy", var_name+"wz"])

			# append all of the dataframes 
			dfs.append(df)

	# concatenate all dataframes in a big dataframe
	return pd.concat(dfs, axis=1)






