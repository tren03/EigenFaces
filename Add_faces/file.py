'create a file if it doesnt exist , if it exists ask the user to provide a longer nane or somehting unique to identify the user'

import os

# Get the current working directory
cwd = os.getcwd()

# Get all the directory names in the current working directory
os.chdir("Faces")
directories= os.listdir()
print(type(directories[0]))

# Print all the directory names
for directory in directories:
    print(directory)