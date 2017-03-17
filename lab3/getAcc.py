import sys

if len(sys.argv) != 2:
	print "Usage: python getAcc.py <FileName>"
	return

with open(sys.argv[1], 'r') as f:
	for line in f:
		line = line.split(":")
		if line[0] == "Mean Squared Error":
			print line[1],
