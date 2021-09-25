
A much more pythonic and elegant way to implement the technique of memoization is through the use of decorators, as
shown below.

```python

def memoize(function, *args):
    cache = {}
    def decorated_function(*args):
        if args in cache:
            print('cached')
            return cache[args]
        else:
            print('not cached')
            val = function(*args)
            cache[args] = val
            return val
    return decorated_function

@memoize
def myfunc(x):
    return x**2

myfunc = np.vectorize(myfunc)

# now there's no need to use a for loop and call the function on each element
# nor is there a need to pass an explicit dictionary to function

```
The code implements mechanism to count function calls using global variable counters.
