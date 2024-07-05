n = 3002

max_i = 0
counter = 0 
for i in range(1, n+1):
    counter += 1
    if i > max_i:
        max_i = i
print(max_i)
print(counter)

print(n-1)
