
err = None
try:
    print(a2)

except Exception as e:
    err = e
    print(e)

print(2)
if err: raise err