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

BOND_DIMS = []

class Tensor:
    """Tensor class for netcon.

    Attributes:
        rpn: contraction sequence with reverse polish notation.
        bit: bit representation of contracted tensors.
        bonds: list of bonds connecting with the outside.
        is_new: a flag.
    """

    def __init__(self,rpn=[],bit=0,bonds=[],cost=0.0,is_new=True):
        self.rpn = rpn[:]
        self.bit = bit
        self.bonds = bonds
        self.cost = cost
        self.is_new = is_new

    def __str__(self):
        return "{0} : bond={1} cost={2:.6e} bit={3}  new={4}".format(
            self.rpn, list(self.bonds), self.cost, self.bit, self.is_new)


def netcon(tn, bond_dims):
    """Find optimal contraction sequence.

    Args:
        tn: TensorNetwork in tdt.py
        bond_dims: List of bond dimensions.

    Return:
        rpn: Optimal contraction sequence with reverse polish notation.
        cost: Total contraction cost.
    """
    BOND_DIMS[0:0] = bond_dims
    tensor_set = _init(tn)

    n = len(tensor_set[0])
    xi_min = float(min(BOND_DIMS))
    mu_cap = 1.0
    mu_old = 0.0

    while len(tensor_set[-1])<1:
        logging.info("netcon: searching with mu_cap={0:.6e}".format(mu_cap))
        mu_next = sys.float_info.max
        for c in range(1,n):
            for d1 in range((c+1)/2):
                d2 = c-d1-1
                n1 = len(tensor_set[d1])
                n2 = len(tensor_set[d2])
                for i1 in range(n1):
                    i2_start = i1+1 if d1==d2 else 0
                    for i2 in range(i2_start, n2):
                        t1 = tensor_set[d1][i1]
                        t2 = tensor_set[d2][i2]

                        if _is_disjoint(t1,t2): continue
                        if _is_overlap(t1,t2): continue

                        mu = _get_cost(t1,t2)
                        mu_0 = 0.0 if (t1.is_new or t2.is_new) else mu_old

                        if (mu > mu_cap) and (mu < mu_next): mu_next = mu
                        if (mu > mu_0) and (mu <= mu_cap):
                            t_new = _contract(t1,t2)
                            is_find = False
                            for i,t_old in enumerate(tensor_set[c]):
                                if t_new.bit == t_old.bit:
                                    if t_new.cost < t_old.cost:
                                        tensor_set[c][i] = t_new
                                    is_find = True
                                    break
                            if not is_find: tensor_set[c].append(t_new)
        mu_old = mu_cap
        mu_cap = max(mu_next, mu_cap*xi_min)
        for s in tensor_set:
            for t in s: t.is_new = False

        logging.debug("netcon: tensor_num=" +  str([ len(s) for s in tensor_set]))

    t_final = tensor_set[-1][0]
    return t_final.rpn, t_final.cost


def _init(tn):
    """Initialize a set of tensors from tdt tensor-network."""
    tensor_set = [[] for t in tn.tensors]
    for t in tn.tensors:
        rpn = t.name
        bit = 0
        for i in rpn:
            if i>=0: bit += (1<<i)
        bonds = frozenset(t.bonds)
        cost = 0.0
        tensor_set[0].append(Tensor(rpn,bit,bonds,cost))
    return tensor_set


def _get_cost(t1,t2):
    """Get the cost of contraction of two tensors."""
    cost = 1.0
    for b in (t1.bonds | t2.bonds):
        cost *= BOND_DIMS[b]
    cost += t1.cost + t2.cost
    return cost


def _contract(t1,t2):
    """Return a contracted tensor"""
    assert (not _is_disjoint(t1,t2))
    rpn = t1.rpn + t2.rpn + [-1]
    bit = t1.bit ^ t2.bit # XOR
    bonds = frozenset(t1.bonds ^ t2.bonds)
    cost = _get_cost(t1,t2)
    return Tensor(rpn,bit,bonds,cost)


def _is_disjoint(t1,t2):
    """Check if two tensors are disjoint."""
    return (t1.bonds).isdisjoint(t2.bonds)


def _is_overlap(t1,t2):
    """Check if two tensors have the same basic tensor."""
    return (t1.bit & t2.bit)>0


def _print_tset(tensor_set):
    """Print tensor_set. (for debug)"""
    for level in range(len(tensor_set)):
        for i,t in enumerate(tensor_set[level]):
            print level,i,t

