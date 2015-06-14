from numpy.random import *
from numpy import *

def r_smpl(w):
  num = len(w)
  index = []
  count = [0.] + [sum(w[:i+1]) for i in range(num)]
  e0, k = random(), 0
  for e in [(e0+i)/num for i in range(num)]:
    while e > count[k]:
      k+=1
    index.append(k-1)
  return index

def par_fil(s, p, step, num):
  sq = iter(s)
  x = ones((num, 2), int) * p
  m0 = sq.next()[tuple(p)] * ones(num)
  yield p, x, ones(num)/num
  for i in sq:
    x = x + uniform(-step, step, x.shape)
    x  = x.clip(zeros(2), array(i.shape)-1).astype(int)
    m  = i[tuple(x.T)]
    wei  = 1./(1. + (m0-m)**2)
    wei /= sum(wei)
    yield sum(x.T*wei, axis=1), x, wei
    if 1./sum(wei**2) < num/2.:
      x  = x[r_smpl(wei),:]               

if __name__ == "__main__":
  from pylab import *
  import time
  from itertools import izip

  ion()
  sequence = [ i for i in zeros((30,300,300), int)]
  x0 = array([150, 150])
  xtr = vstack((arange(20)*3, arange(20)*2)).T + x0
  for ti, x in enumerate(xtr):
    x_slice = slice(x[0]-8, x[0]+8)
    y_slice = slice(x[1]-8, x[1]+8)
    sequence[ti][x_slice, y_slice] = 255

  for i, p in izip(sequence, par_fil(sequence, x0, 8, 100)):
    ps, xtr, wtr = p
    pos_ovlay = zeros_like(i)
    pos_ovlay[tuple(ps)] = 1
    par_ovlay = zeros_like(i)
    par_ovlay[tuple(xtr.T)] = 1
    hold(True)
    draw()
    time.sleep(0.3)
    clf()
    imshow(i,cmap=cm.gray)
    spy(pos_ovlay, marker='.', color='r')
    spy(par_ovlay, marker=',', color='w')
  show()
