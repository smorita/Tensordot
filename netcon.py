"""Get optimal contraction sequence using netcon algorithm

Reference:
    R. N. C. Pfeifer, et al.: Phys. Rev. E 90, 033315 (2014)
"""

__author__ = "Satoshi MORITA <morita@issp.u-tokyo.ac.jp>"
__date__ = "24 March 2016"

import sys
import logging
import time
import config


class HistTensorFrame:
    """Tensor class for netcon.

    Attributes:
        rpn: contraction sequence with reverse polish notation.
        bits: bits representation of contracted tensors.
        bonds: list of uncontracted bonds.
        is_new: a flag.
    """

    def __init__(self,rpn=[],bits=0,bonds=[],cost=0.0,is_new=True):
        self.rpn = rpn[:]
        self.bits = bits
        self.bonds = bonds
        self.cost = cost
        self.is_new = is_new

    def __repr__(self):
        return "HistTensorFrame({0}, bonds={1}, cost={2:.6e}, bits={3}, is_new={4})".format(
            self.rpn, self.bonds, self.cost, self.bits, self.is_new)

    def __str__(self):
        return "{0} : bonds={1} cost={2:.6e} bits={3} new={4}".format(
            self.rpn, self.bonds, self.cost, self.bits, self.is_new)


class NetconClass:
    def __init__(self, prime_tensors, bond_dims):
        #print(tensors)
        self.prime_tensors = prime_tensors
        self.BOND_DIMS = bond_dims[:]

    def calc(self):
        """Find optimal contraction sequence.

        Args:
            tn: TensorNetwork in tdt.py
            bond_dims: List of bond dimensions.

        Return:
            rpn: Optimal contraction sequence with reverse polish notation.
            cost: Total contraction cost.
        """
        tensors_of_size = self._init_tensors_of_size()

        n = len(self.prime_tensors)
        xi_min = float(min(self.BOND_DIMS))
        mu_cap = 1.0
        mu_old = 0.0

        while len(tensors_of_size[-1])<1:
            logging.info("netcon: searching with mu_cap={0:.6e}".format(mu_cap))
            mu_next = sys.float_info.max
            for c in range(1,n):
                cp1 = c+1
                for d1 in range((c+1)//2):
                    d1p1 = d1+1
                    d2 = c-d1p1
                    d2p1 = d2+1
                    n1 = len(tensors_of_size[d1p1])
                    n2 = len(tensors_of_size[d2p1])
                    for i1 in range(n1):
                        i2_start = i1+1 if d1p1==d2p1 else 0
                        for i2 in range(i2_start, n2):
                            t1 = tensors_of_size[d1p1][i1]
                            t2 = tensors_of_size[d2p1][i2]

                            if self._is_disjoint(t1,t2): continue
                            if self._is_overlap(t1,t2): continue

                            mu = self._get_cost(t1,t2)
                            mu_0 = 0.0 if (t1.is_new or t2.is_new) else mu_old

                            if (mu > mu_cap) and (mu < mu_next): mu_next = mu
                            if (mu > mu_0) and (mu <= mu_cap):
                                t_new = self._contract(t1,t2)
                                is_find = False
                                for i,t_old in enumerate(tensors_of_size[cp1]):
                                    if t_new.bits == t_old.bits:
                                        if t_new.cost < t_old.cost:
                                            tensors_of_size[cp1][i] = t_new
                                        is_find = True
                                        break
                                if not is_find: tensors_of_size[cp1].append(t_new)
            mu_old = mu_cap
            mu_cap = max(mu_next, mu_cap*xi_min)
            for s in tensors_of_size:
                for t in s: t.is_new = False

            logging.debug("netcon: tensor_num=" +  str([ len(s) for s in tensors_of_size]))

        t_final = tensors_of_size[-1][0]
        #print(t_final.rpn)
        return t_final.rpn, t_final.cost


    def _init_tensors_of_size(self):
        """tensors_of_size[k] == calculated tensors which is contraction of k+1 prime tensors"""
        tensors_of_size = [[] for size in range(len(self.prime_tensors)+1)]
        for t in self.prime_tensors:
            rpn = t.name
            bits = 0
            for i in rpn:
                if i>=0: bits += (1<<i)
            bonds = frozenset(t.bonds)
            cost = 0.0
            tensors_of_size[1].append(HistTensorFrame(rpn,bits,bonds,cost))
        return tensors_of_size


    def _get_cost(self,t1,t2):
        """Get the cost of contraction of two tensors."""
        cost = 1.0
        for b in (t1.bonds | t2.bonds):
            cost *= self.BOND_DIMS[b]
        cost += t1.cost + t2.cost
        return cost


    def _contract(self,t1,t2):
        """Return a contracted tensor"""
        assert (not self._is_disjoint(t1,t2))
        rpn = t1.rpn + t2.rpn + [-1]
        bits = t1.bits ^ t2.bits # XOR
        bonds = frozenset(t1.bonds ^ t2.bonds)
        cost = self._get_cost(t1,t2)
        return HistTensorFrame(rpn,bits,bonds,cost)


    def _is_disjoint(self,t1,t2):
        """Check if two tensors are disjoint."""
        return (t1.bonds).isdisjoint(t2.bonds)


    def _is_overlap(self,t1,t2):
        """Check if two tensors have the same basic tensor."""
        return (t1.bits & t2.bits)>0


    def _print_tset(self,tensors_of_size):
        """Print tensors_of_size. (for debug)"""
        for level in range(len(tensors_of_size)):
            for i,t in enumerate(tensors_of_size[level]):
                print(level,i,t)

