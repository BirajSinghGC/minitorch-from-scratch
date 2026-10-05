"""Collection of the core mathematical operators used throughout the code base."""

from ast import Call
import enum
import math

# ## Task 0.1
from pickletools import floatnl
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return x * y
# TODO fill in the remaining mathematical operators. Use mul provided above as an example.
# - id
def id(x: float) -> float:
    """$f(x) = x$"""
    return x
# - add
def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y
# - neg
def neg(x:float) -> float:
    """$f(x) = -x$"""
    return -x
# - lt
def lt(x:float, y:float) -> float:
    """$f(x, y) = x < y$"""
    return 1.0 if x < y else 0.0
# - eq
def eq(x:float, y:float) -> float:
    """$f(x, y) = x == y$"""
    return 1.0 if x == y else 0.0
# - max
def max(x:float, y:float) -> float:
    """$f(x, y) = max(x, y)$"""
    return x if x > y else y
# - is_close
def is_close(x:float, y:float) -> bool:
    """$f(x, y) = |x - y| < 1e-2$"""
    return abs(x - y) < 1e-2
# - sigmoid
def sigmoid(x:float) -> float:
    # $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
# - relu
def relu(x:float) -> float:
    """$f(x) = max(0, x)$"""
    return max(0, x)
# - log
def log(x:float) -> float:
    """$f(x) = log(x)$"""
    return math.log(x)
# - exp
def exp(x:float) -> float:
    """$f(x) = e^x$"""
    return math.exp(x)
# - log_back
def log_back(x:float, y:float) -> float:
    """$f(x, y) = y / x$"""
    return y / x
# - inv
def inv(x:float) -> float:
    """$f(x) = 1.0 / x$"""
    return 1.0 / x
# - inv_back
def inv_back(x:float, y:float) -> float:
    """$f(x, y) = -y / x^2$"""
    return -y / x**2
# - relu_back
def relu_back(x:float, y:float) -> float:
    """$f(x, y) = y if x > 0 else 0$"""
    if x > 0:
        return y
    else:
        return 0.0
# - exp_back
def exp_back(x:float, y:float) -> float:
    """$f(x, y) = y * e^x$"""
    return y * math.exp(x)

# - sigmoid_back


# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    def apply(x: Iterable[float]) -> Iterable[float]:
        updated_x = []
        for x_value in x:
            updated_x.append(fn(x_value))
        return updated_x
    return apply


def zipWith(fn: Callable[[float, float], float]) -> Callable[ [Iterable[float], Iterable[float]], Iterable[float]]:
    def apply(x: Iterable[float], y: Iterable[float]) -> Iterable[float]:
        zip_xy = []
        for i, j in zip(x,y):
            zip_xy.append(fn(i,j))
        return zip_xy
    return apply

# - reduce
def reduce(fn: Callable[[float, float], float], starting_result: float) -> Callable[[Iterable[float]], float]:
    def apply(x: Iterable[float]) -> float:
        result = starting_result
        for value in x:
                result = fn(result, value)
        return result
    
    return apply
            
                
    
#
# Use these to implement
# - negList : negate all elemnts in a list using map
negList = map(neg)
# - addLists : add corresponding elements from two lists using zipWith
addLists = zipWith(add)
# - sum: sum all elements in a list using reduce
sum = reduce(add, 0)
# - prod: tcalculate the product of all elements in a list using reduce
prod = reduce(mul, 1)

