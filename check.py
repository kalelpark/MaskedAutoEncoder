import time

temp = time.gmtime()
tar = map(lambda x : str(x), list(temp)[:-9])
stwrite = "_".join(tar)
print(stwrite)
# aka = list(temp)
# stwrite = "_".join(aka)
# print(stwrite)