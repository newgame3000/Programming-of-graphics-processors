#!/usr/bin/env python3
 
import numpy as np

n = int(input())

m = np.zeros((n, n))
l = np.zeros((n, n))
u = np.zeros((n, n))
p = np.zeros(n)


for i in range(n):
	for j, v in enumerate(input().split()):
		m[i][j] = float(v)


for i in range(n):
	for j, v in enumerate(input().split()):
		if j >= i:
			u[i][j] = float(v)
		else:
			l[i][j] = float(v)
		if i == j:
			l[i][j] = 1

for j, v in enumerate(input().split()):
	p[j] = int(v)

res = l @ u

#print(res)

for i in range(n - 1, -1, -1):
	for j in range(n):
		temp = res[i][j]
		res[i][j] = res[int(p[i])][j]
		res[int(p[i])][j] = temp



	#print(res)
good = True

for i in range(n):
	for j in range(n):
		if abs(m[i][j] - res[i][j]) >= 1e-4:
			good = False
			break 

if good:
	print("OK")
else:
	print("Error")
	print(m)
	print("\n")
	print(res)
