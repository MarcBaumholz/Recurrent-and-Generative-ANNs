#! /usr/bin/env python3

import numpy as np
import os
import sys

sys.path.append("../utils")
import helper_functions as helpers


#####################
# PUBLIC PARAMETERS #
#####################

DATASET_NAME = "chapter1"


#############
# FUNCTIONS #
#############


def txt_to_npy(path):
	"""
	Creates an alphabet and sequences of all lines from a given .txt file and
	writes the one-hot vector sequences to separate files.
	:param path: The path to the .txt file containing the raw text
	"""

	# Load the .txt file
	with open(path, "r", encoding="utf-8") as file:

		#
		# Determine the alphabet of the text

		# List to store all characters of the text
		chars = []

		# Append all characters to the chars list
		for line in file:
			if len(line) <= 1:
				continue

			for char in line:
				# Only consider lower-case characters
				chars.append(char.lower())

		# Determine the unique characters and write them to file
		alphabet = np.unique(np.array(chars))
		np.save(os.path.join(DATASET_NAME, "alphabet.npy"), alphabet)

		#
		# Create one-hot vector sequences

		# Reset the file pointer to the first position
		file.seek(0, 0)

		data_idx = 0

		# Iterate over all lines and create a one-hot sequence from them
		for line in file:
			if len(line) <= 1:
				continue

			# TODO: Convert the line into a sequence of one hot vectors and
			#		write it to file as sample_****.npy, using np.save as
			#		above. You may implement and use a char_to_one_hot method
			#		in the helper_functions.py (utils directory).
			one_hot_sequence = np.array([helpers.char_to_one_hot(char, alphabet) for char in line])
			filename = os.path.join(DATASET_NAME, f"sample_{data_idx}.npy")
			np.save(filename, one_hot_sequence)
			data_idx += 1


##########
# SCRIPT #
##########

def main():
	os.makedirs(DATASET_NAME, exist_ok=True)
	txt_to_npy("text.txt")

if __name__ == "__main__":
	main()