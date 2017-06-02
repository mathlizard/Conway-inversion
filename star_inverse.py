"""
Computes the inverse of a matrix using the Conway star matrix formula. 
call it with `star_inverse(m)`
where `m` is a (square) numpy matrix
"""
import numpy
from numpy import dot
import time


def star(m, k=1):
    """ recursive way of computing star """
    if (m.shape == (1,1)):
        return numpy.matrix([[1. / (1. - m[0,0])]])
    r = numpy.zeros(shape=m.shape)
    a_star = star(m[:k,:k])
    d_star = star(m[k:,k:])
    r[:k,:k] = star(m[:k,:k] + dot(dot(m[:k,k:], d_star), m[k:,:k]))
    r[:k,k:] = dot(dot(r[:k,:k], m[:k,k:]), d_star)
    r[k:,k:] = star(m[k:,k:] + dot(dot(m[k:,:k], a_star), m[:k,k:]))
    r[k:,:k] = dot(dot(r[k:,k:], m[k:,:k]), a_star)
    return r


def star_inverse(m):
    """ returns the inverse of m """
    if (m.shape[0] != m.shape[1]):
        raise ValueError("m must be a square matrix! ")
    return star(numpy.eye(m.shape[0]) - m)


def test(m):
    """ simple test function ~ 
        right now it measures function call overhead on small matrices """
    print("-" * 80)
    print("matrix:")
    print(m)
    ei = abs(numpy.linalg.eigvals(m))
    print("maximum absolute eigenvalue = " + str(max(ei)))
    print("minimum absolute eigenvalue = " + str(min(ei)))
    print()
    print("star inverse : ")
    t = time.time()
    print(star_inverse(m))
    print("time: " + str(time.time() - t))
    print()
    print("numpy.linalg.inv : ")
    try:
        t = time.time()
        print(numpy.linalg.inv(m))
        print("time: " + str(time.time() - t))
    except:
        print("linalg: singular matrix")


def main():
    """ call this to test the matrix inverse """
    test(numpy.matrix([[1,2], [3,4]]))
    test(numpy.matrix([[3,2,3], [4,5,3], [7,8,9]]))
    test(numpy.matrix([[1,2,3], [4,5,6], [7,8,9]]))
    test(numpy.matrix([[1,9,9], [10,1,3], [10,9,5]]))


if __name__ == '__main__':
    main()
