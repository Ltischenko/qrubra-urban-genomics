#!/usr/bin/python3
# -*- coding: iso-8859-1 -*-

# Import necessary libraries
import pandas as pd
import sys, fileinput


# Verify arguments
try:
    genot = sys.argv[1]
    id1 = sys.argv[2]
    metad = sys.argv[3]
    id2 = sys.argv[4]
    out = sys.argv[5]

except:
    print('''
          Something is missing.
          
          Usage:
            python make_input.py geno var1 metadata var2
        
            var1 and 2 are variable names for the ids in each file.
            Those can be empty with "" if necessary but it is advised to
            fill up the sample ID columns with a header
        ''')
    sys.exit(1)

# Read the genotype file into a DataFrame
geno = pd.read_csv(genot)
print("\n\nHere's the preview of the genotypes:")
print(geno)

# Read the metadata file into a DataFrame
meta = pd.read_csv(metad)
print("\n\nHere's the preview of the metadata:")
print(meta)

# fill empty values
if id1 == "":
  id1 = "Unnamed: 0"

if id2 == "":
  id2 = "Unnamed: 0"

# Merge the genotype and metadata DataFrames on the specified ID columns
data = pd.merge(geno, meta, left_on=id1, right_on=id2)
print("\n\nHere's the preview of the input file:")
print(data)

# Print the dimensions of the merged DataFrame
print("\n\ninput dimensions:")
print(data.shape)

# Print rows where the IDs do not match
print("\n\nrows with different ids:")
print(data[data[id1] != data[id2]])

# Drop the ID column from the metadata file
data.drop([id2],axis=1)

# Change name of the first column
data.rename(columns={'Unnamed: 0': 'Samples'}, inplace=True)

# Save the merged DataFrame to a CSV file
data.to_csv(out+'.csv', index=False)

print("\n\nCONGRATS; input saved")