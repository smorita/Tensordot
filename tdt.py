#!/usr/bin/env python
import sys
import argparse
import logging
import time
import random
import config

# Global variables
TENSOR_NAMES = []
BOND_NAMES = []
BOND_DIMS = []
FINAL_ORDER = None
INFO_TIME_LIMIT = [False, 0, 0]

class Tensor:
    def __init__(self,name=None,bonds=[]):
        if name==None:
            self.name = []
        elif isinstance(name, list):
            self.name = name[:]
        else:
            self.name = [name]
        self.bonds = bonds[:]

    def __str__(self):
        return str(self.name) + ", " + str(self.bonds)

class Bond:
    def __init__(self,t0=-1,t1=-1):
        self.t0 = t0
        self.t1 = t1

    def __str__(self):
        return "({0},{1})".format(self.t0,self.t1)

    def isFree(self):
        return (self.t0 < 0 or self.t1 < 0)

    def connect(self,tensor_index):
        assert self.isFree(), "edge already connected to two tensors"
        if self.t0<0:
            self.t0 = tensor_index
        else:
            assert not self.t0==tensor_index, "edge connects to the same tensor"
            self.t1 = tensor_index

    def has(self,tensor_index):
        return (self.t0==tensor_index or self.t1==tensor_index)

class TensorNetwork:
    def __init__(self):
        self.tensors = []
        self.bonds = []
        self.total_memory = 0.0
        self.max_memory = 0.0
        self.cpu_cost = 0.0
        self.visited_bonds = []

    def __str__(self):
        s = ""
        for i,t in enumerate(self.tensors):
            s += "tensor {0} : {1}\n".format(i,t)
        for i,b in enumerate(self.bonds):
            s += "bond {0} : {1}, {2} {3}\n".format(i,BOND_NAMES[i],b,BOND_DIMS[i])
        s += "memory : {0}\n".format(self.total_memory)
        s += "cpu : {0}\n".format(self.cpu_cost)
        return s


    def init(self):
        self.calculate_memory()
        self.output_log("input")


    def clone(self):
        tn = TensorNetwork()
        tn.total_memory = self.total_memory
        tn.max_memory = self.max_memory
        tn.cpu_cost = self.cpu_cost
        tn.visited_bonds = self.visited_bonds[:]
        tn.bonds = [ Bond(b.t0,b.t1) for b in self.bonds ]
        tn.tensors = [ Tensor(t.name,t.bonds) for t in self.tensors ]
        return tn


    def output_log(self,prefix=""):
        if not prefix=="": prefix += " "
        for i,t in enumerate(self.tensors):
            logging.info(prefix + "tensor{0} : {1} {2}".format(i,TENSOR_NAMES[i],t.bonds))
        for i,b in enumerate(self.bonds):
            logging.info(prefix + "bond{0} : {1} {2} {3}".format(i,BOND_NAMES[i],b,BOND_DIMS[i]))
        logging.info(prefix + "memory : {0}".format(self.total_memory))
        logging.info(prefix + "cpu : {0}".format(self.cpu_cost))


    def add_tensor(self, t_name, b_names):
        t_index = len(self.tensors)
        b_indexs = []
        for b in b_names:
            if b not in BOND_NAMES:
                self.bonds.append(Bond())
                BOND_NAMES.append(b)
                BOND_DIMS.append(config.DEFAULT_BOND_DIM)

            i = BOND_NAMES.index(b)
            self.bonds[i].connect(t_index)
            b_indexs.append(i)

        TENSOR_NAMES.append(t_name)
        self.tensors.append(Tensor(t_index,b_indexs))


    def clear_visited_bonds(self):
        self.visited_bonds = [ False ] * len(self.bonds)


    def calculate_memory(self):
        mem = 0.0
        for t in self.tensors:
            val = 1.0
            for b in t.bonds: val *= BOND_DIMS[b]
            mem += val
        self.total_memory = mem
        self.max_memory = mem


    def find_bonds(self, tensor_a, tensor_b):
        bonds_a = frozenset(self.tensors[tensor_a].bonds)
        bonds_b = frozenset(self.tensors[tensor_b].bonds)
        return list(bonds_a & bonds_b), list(bonds_a - bonds_b), list(bonds_b - bonds_a)


    def safe_find_bonds(self, tensor_a, tensor_b):
        bonds_a = self.tensors[tensor_a].bonds
        bonds_b = self.tensors[tensor_b].bonds
        contract = [ b for b in bonds_a if b in bonds_b]
        replaced_a = [ b for b in bonds_a if b not in bonds_b ]
        replaced_b = [ b for b in bonds_b if b not in bonds_a ]
        return contract, replaced_a, replaced_b


    def find_bonds_from_bond(self, bond_index):
        bond = self.bonds[bond_index]
        return self.find_bonds(bond.t0, bond.t1)

    def calculate_costs(self, bc, br0, br1):
        """caluculate cpu cost and memory change"""
        dc = d0 = d1 = 1.0
        for b in bc: dc *= BOND_DIMS[b]
        for b in br0: d0 *= BOND_DIMS[b]
        for b in br1: d1 *= BOND_DIMS[b]
        cpu = dc*d0*d1
        mem_add = d0*d1
        mem_reduce = (d0+d1)*dc

        return cpu, mem_add, mem_reduce

    def judge_contract(self, cpu, mem_add, min_cpu, min_mem):
        max_memory = max(self.total_memory+mem_add, self.max_memory)
        cpu_cost = self.cpu_cost + cpu

        # The case that max_memory==min_mem is accept
        judge = (cpu_cost < min_cpu-0.5) and (max_memory < config.MEMORY_ACCEPTABLE_RATIO*min_mem+0.5)

        return judge


    def contract(self, t0, t1, bc, br0, br1):
        tn = self.clone()

        # create the contracted tensor
        t_new = tn.tensors[t0]
        ## change names of tensors using Reverse Polish Notation
        t_new.name = self.tensors[t0].name+self.tensors[t1].name+[-1]
        ## remove contracted bonds
        for b in bc: t_new.bonds.remove(b)
        ## add bonds from deleted tensor
        for b in br1: t_new.bonds.append(b)

        # clear the removed tensor
        tn.tensors[t1] = Tensor()

        # update bonds
        bonds = tn.bonds
        ## remove contracted bonds from the bond list
        for b in bc: bonds[b].t0 = bonds[b].t1 = -1
        ## change bond connections
        old_idx = t1
        new_idx = t0
        for b in br1:
            if bonds[b].t0==old_idx: bonds[b].t0=new_idx
            elif bonds[b].t1==old_idx: bonds[b].t1=new_idx

        tn.clear_visited_bonds()
        return tn


    def contract_bond(self, bond_index, bc, br0, br1):
        bond = self.bonds[bond_index]
        return self.contract(bond.t0, bond.t1, bc, br0, br1)


    def update_costs(self, cpu, mem_add, mem_reduce):
        # update total memory and cpu cost
        self.max_memory = max(self.total_memory + mem_add, self.max_memory)
        self.total_memory += mem_add - mem_reduce
        self.cpu_cost += cpu


    def judge_accept(self, min_cpu, min_mem):
        # judge by cpu cost
        if self.cpu_cost < min_cpu-0.5: return True
        elif self.cpu_cost > min_cpu+0.5: return False

        # Now cpu cost is same, then check memory usage.
        if self.max_memory < min_mem-0.5: return True

        return False

def set_bond_dim(bond_name, dim):
    BOND_DIMS[ BOND_NAMES.index(bond_name) ] = dim

def string_stack(stack):
    return "".join( [str(s[0]) for s in stack] )

def find_path(tn_orig):
    global INFO_TIME_LIMIT
    min_cpu = sys.float_info.max
    min_mem = sys.float_info.max
    max_level=len(tn_orig.tensors)-1

    tn_tree = [[] for i in range(max_level+1)]
    tn_tree[0] = tn_orig.clone()

    tn_tree[0].clear_visited_bonds()

    stack = []
    for i,b in enumerate(tn_tree[0].bonds):
        if not b.isFree(): stack.append((0,i))
    INFO_TIME_LIMIT[2] = len(stack)

    count = -1
    count_find = 0
    start_time = restart_time = time.time()

    while(len(stack)>0):
        level,bond = stack.pop()
        tn = tn_tree[level]
        count += 1
        log_prefix = "{0} ({1},{2}) ".format(count,level,bond)

        # logging
        now = time.time()
        if now-restart_time > config.INFO_INTERVAL:
            restart_time = now
            logging.info(log_prefix+"{0} min_cpu={1:.6e} min_mem={2:.6e}".format(
                string_stack(stack), min_cpu, min_mem))
        elif config.LOGGING_LEVEL == logging.DEBUG:
            logging.debug(log_prefix+"{0} min_cpu={1:.6e} min_mem={2:.6e}".format(
                string_stack(stack), min_cpu, min_mem))
        if now-start_time > config.TIME_LIMIT:
            logging.warning(log_prefix+"Stopped by time limit.")
            INFO_TIME_LIMIT[0] = True
            INFO_TIME_LIMIT[1] = len([s for s in stack if s[0]==0])+1
            stack = []
            continue


        # bond is already visited.
        if tn.visited_bonds[bond]:
            logging.debug(log_prefix+"Bond is already visited.")
            continue

        # Get contracted bond, replaced bonds
        bc, br0, br1 = tn.find_bonds_from_bond(bond)

        # Update visited bonds
        for b in bc: tn.visited_bonds[b] = True

        # Caluculate cpu cost and memory change
        cpu, mem_add, mem_reduce = tn.calculate_costs(bc, br0, br1)

        # judge, bc, br0, br1, cpu, mem_add, mem_reduce = tn.judge_contract(bond, min_cpu, min_mem)
        if not tn.judge_contract(cpu, mem_add, min_cpu, min_mem):
            logging.debug(log_prefix+"Cut branch.")
            continue

        # tn_tree[level+1] = tn.contract(bond, bc, br0, br1, cpu, mem_add, mem_reduce)
        tn_tree[level+1] = tn.contract_bond(bond, bc, br0, br1)

        tn_tree[level+1].update_costs(cpu, mem_add, mem_reduce)

        # all tensors has been contracted
        if level+1==max_level:
            tn = tn_tree[level+1]
            if tn.judge_accept(min_cpu,min_mem):
                min_cpu = tn.cpu_cost
                min_mem = tn.max_memory
                # get script from the name of the contracted tensor
                for t in tn.tensors:
                    if not t.name==[]:
                        script = t.name
                        break
                count_find = count
                logging.info(log_prefix+"Find script={0} cpu={1:.6e} mem={2:.6e}".format(
                    script, tn.cpu_cost, tn.max_memory))
            else:
                logging.debug(log_prefix+"Reject script={0} cpu={1:.6e} mem={2:.6e}".format(
                    script, tn.cpu_cost, tn.max_memory))
            continue

        # add inner bonds in the next level into the stack
        stack.extend( next_items(tn_tree[level+1].bonds,br0,br1,bond,level+1) )

        logging.debug(log_prefix+"Go to the next level. next_bonds={0}".format(
            len(tn_tree[level+1].bonds)))

    logging.info(log_prefix+"Finish {0}/{1} script={2}".format(count_find, count, script))
    return script, min_cpu, min_mem


def next_items(bonds,br0,br1,bond,next_level):
    """Create list of inner bonds to be checked in the next level"""
    # elements in frozenset satisfy  i<bond or (i in br0) or (i in br1)
    return [ (next_level,i) for i in (frozenset(range(bond) + br0 + br1)) \
             if not bonds[i].isFree() ]


def random_search(tn_orig,iteration):
    global INFO_TIME_LIMIT

    min_cpu = sys.float_info.max
    min_mem = sys.float_info.max
    max_level=len(tn_orig.tensors)-1

    tn_tree = [[] for i in range(max_level+1)]
    tn_tree[0] = tn_orig.clone()
    tn_tree[0].clear_visited_bonds()

    script = ""
    count_find = 0
    start_time = restart_time = time.time()

    for count in range(iteration):
        for level in range(max_level):
            tn = tn_tree[level]

            inner_bonds = []
            for i,b in enumerate(tn.bonds):
                if not b.isFree(): inner_bonds.append(i)
            bond = random.choice(inner_bonds)

            # logging
            log_prefix = "{0} ({1},{2}) ".format(count,level,bond)
            now = time.time()
            if now-restart_time > config.INFO_INTERVAL:
                restart_time = now
                logging.info(log_prefix+"min_cpu={0:.6e} min_mem={1:.6e}".format(
                    min_cpu, min_mem))
            elif config.LOGGING_LEVEL == logging.DEBUG:
                logging.debug(log_prefix+"min_cpu={0:.6e} min_mem={1:.6e}".format(
                    min_cpu, min_mem))


            # Get contracted bond, replaced bonds
            bc, br0, br1 = tn.find_bonds_from_bond(bond)

            # Caluculate cpu cost and memory change
            cpu, mem_add, mem_reduce = tn.calculate_costs(bc, br0, br1)

            # judge, bc, br0, br1, cpu, mem_add, mem_reduce = tn.judge_contract(bond, min_cpu, min_mem)
            if not tn.judge_contract(cpu, mem_add, min_cpu, min_mem):
                logging.debug(log_prefix+"Cut branch.")
                break

            tn_tree[level+1] = tn.contract_bond(bond, bc, br0, br1)
            tn_tree[level+1].update_costs(cpu, mem_add, mem_reduce)

            logging.debug(log_prefix+"Go to the next level. next_bonds={0}".format(
                len(tn_tree[level+1].bonds)))

        # all tensors has been contracted
        if level+1==max_level:
            tn = tn_tree[level+1]
            if tn.judge_accept(min_cpu,min_mem):
                min_cpu = tn.cpu_cost
                min_mem = tn.max_memory
                # get script from the name of the contracted tensor
                for t in tn.tensors:
                    if not t.name==[]:
                        script = t.name
                        break
                count_find = count
                logging.info(log_prefix+"Find script={0} cpu={1:.6e} mem={2:.6e}".format(
                    script, tn.cpu_cost, tn.max_memory))
            else:
                logging.debug(log_prefix+"Reject script={0} cpu={1:.6e} mem={2:.6e}".format(
                    script, tn.cpu_cost, tn.max_memory))

        if now-start_time > config.TIME_LIMIT:
            logging.warning(log_prefix+"Stopped by time limit.")
            INFO_TIME_LIMIT[0] = True
            INFO_TIME_LIMIT[1] = count
            INFO_TIME_LIMIT[2] = iteration
            print INFO_TIME_LIMIT
            break

    logging.info(log_prefix+"Finish {0}/{1} script={2}".format(count_find, count, script))
    return script, min_cpu, min_mem


def get_math(rpn):
    """Generate mathematical formula from Reverse Polish Notation"""
    stack = []
    for c in rpn:
        if c==-1:
            t1 = stack.pop()
            t0 = stack.pop()
            new_name = "("+t0+"*"+t1+")"
            stack.append( new_name )

        else:
            stack.append(TENSOR_NAMES[c])
    return stack[0]

def get_script(tn_orig,rpn):
    """Generate tensordot script from Reverse Polish Notation"""
    tn = tn_orig.clone()
    index = []
    name = []
    for c in rpn:
        if c==-1:
            index1 = index.pop()
            index0 = index.pop()
            name1 = name.pop()
            name0 = name.pop()

            t0 = tn.tensors[index0]
            t1 = tn.tensors[index1]

            bc, br0, br1 = tn.safe_find_bonds(index0, index1)

            axes0 = [ t0.bonds.index(b) for b in bc]
            axes1 = [ t1.bonds.index(b) for b in bc]

            tn = tn.contract(index0, index1, bc, br0, br1)

            trace = (len(br0)==0 and len(br1)==0)
            new_name = tensordot_script(name0,name1,axes0,axes1,trace)

            index.append(index0)
            name.append(new_name)

        else:
            index.append(c)
            name.append([TENSOR_NAMES[c]])

    bond_order = tn.tensors[index.pop()].bonds

    return name.pop(), bond_order


def tensordot_script(name0,name1,axes0,axes1,trace=False):
    if config.STYLE == "numpy":
        func_name = config.NUMPY+".tensordot"
        axes = "(" + str(axes0) + ", " + str(axes1) + ")"
    elif config.STYLE == "mptensor":
        func_name = "trace" if trace else "tensordot"
        str_axes0 = str(tuple(axes0)) if len(axes0)>1 else "("+str(axes0[0])+")"
        str_axes1 = str(tuple(axes1)) if len(axes1)>1 else "("+str(axes1[0])+")"
        axes = "Axes" + str_axes0 + ", " + "Axes" + str_axes1

    script = []
    script.append( func_name + "(" )
    for l in name0: script.append(config.INDENT + l)
    script[-1] += ", " + name1[0]
    for i in range(1,len(name1)): script.append(config.INDENT + name1[i])
    script[-1] += ", " + axes
    script.append( ")" )

    return script


def transpose_script(name,axes):
    if config.STYLE == "numpy":
        func_name = config.NUMPY+".transpose"
        axes = str(axes)
    elif config.STYLE == "mptensor":
        func_name = "transpose"
        str_axes = str(tuple(axes)) if len(axes)>1 else "("+str(axes[0])+")"
        axes = "Axes" + str_axes

    script = []
    script.append( func_name + "(" )
    for l in name: script.append(config.INDENT + l)
    script[-1] += ", " + axes
    script.append( ")" )

    return script


def add_transpose(tn,script,bond_order):
    if FINAL_ORDER == None: return script, bond_order

    f_order = [ BOND_NAMES.index(b) for b in FINAL_ORDER ]

    if not sorted(f_order)==sorted(bond_order):
        logging.warning("The final bond order is invalid. It is ignored.")
        return script,bond_order
    elif f_order == bond_order:
        logging.info("The final bond order was requested, but Transpose is not necessary.")
        return script, bond_order

    axes = [ bond_order.index(b) for b in f_order ]
    return transpose_script(script,axes), f_order


def set_style(style):
    if style=="numpy":
        config.STYLE = "numpy"
        config.COMMENT_PREFIX = "#"
    elif style=="mptensor":
        config.STYLE = "mptensor"
        config.COMMENT_PREFIX = "//"

def set_time_limit(time_limit):
    if time_limit is None:
        pass
    elif time_limit>0.0:
        config.TIME_LIMIT = time_limit
    else:
        config.TIME_LIMIT = sys.float_info.max

def read_file(infile, tn):
    """Read input file"""
    global FINAL_ORDER

    for line in infile:
        data = line.split()
        if data==[]: continue

        command = data[0].lower()
        if command=="style":
            set_style(data[1].lower())
        elif command=="numpy":
            config.NUMPY = data[1]
        elif command=="indent":
            config.INDENT = " " * int(data[1])
        elif command=="time_limit":
            config.TIME_LIMIT = float(data[1])
        elif command=="info_interval":
            config.INFO_INTERVAL = float(data[1])
        elif command=="default_dimension":
            # Should be set the top of input file.
            config.DEFAULT_BOND_DIM = int(data[1])
        elif command=="memory_acceptable_ratio":
            config.MEMORY_ACCEPTABLE_RATIO = max(float(data[1]), 1.0)
        elif command=="debug":
            config.LOGGING_LEVEL = logging.DEBUG

        elif command=="tensor":
            tn.add_tensor(data[1], data[2:])
        elif command=="bond":
            for b in data[1:-1]: set_bond_dim(b, int(data[-1]))
        elif command=="bond_dim":
            for b in data[2:]: set_bond_dim(b, int(data[1]))
        elif command=="order":
            FINAL_ORDER = data[1:]
    infile.close()


def output_result(outfile,script,math_script,cpu,mem,final_bonds,input_file):
    BR = "\n"
    SP = " "
    output = [config.COMMENT_PREFIX*30,
              config.COMMENT_PREFIX + SP + input_file,
              config.COMMENT_PREFIX*30,
              config.COMMENT_PREFIX + SP + math_script,
              config.COMMENT_PREFIX + SP + "cpu_cost= {0:g}  memory= {1:g}".format(cpu, mem),
              config.COMMENT_PREFIX + SP + "final_bond_order " + final_bonds,
              config.COMMENT_PREFIX*30]
    if INFO_TIME_LIMIT[0]:
        num, den = (INFO_TIME_LIMIT[2]-INFO_TIME_LIMIT[1]), INFO_TIME_LIMIT[2]
        ratio = 100.0*num/den
        info = "{0:3.2f}% ({1:d}/{2:d})".format(ratio,num,den)
        output += [config.COMMENT_PREFIX + SP + "Stopped by time limit. " + info + " was finished.",
                   config.COMMENT_PREFIX*30]
    output += script
    outfile.write(BR.join(output) + BR)

def check_bond_order(tn):
    return FINAL_ORDER == None or \
        frozenset(FINAL_ORDER) == frozenset( BOND_NAMES[i] for i,b in enumerate(tn.bonds) if b.isFree() )

def main(args,rand_flag=False):
    tn = TensorNetwork()

    # Read input file
    read_file(args.infile, tn)

    # Overwrite by command-line option
    set_style(args.style)
    set_time_limit(args.time_limit)

    assert len(tn.tensors)>0, "No tensor."
    assert len(tn.bonds)>0, "No bond."
    assert check_bond_order(tn) , "Final bond order is invalid."
    logging.basicConfig(format="%(levelname)s:%(message)s", level=config.LOGGING_LEVEL)

    tn.init()

    if rand_flag:
        rpn, cpu, mem = random_search(tn, max(1,args.iteration))
    else:
        rpn, cpu, mem = find_path(tn)
    script, bond_order = get_script(tn, rpn)
    script, bond_order = add_transpose(tn, script, bond_order)

    final_bonds = "(" + ", ".join([BOND_NAMES[b] for b in bond_order]) + ")"

    output_result(args.outfile,
                  script,get_math(rpn),cpu,mem,final_bonds,
                  args.infile.name)

    if INFO_TIME_LIMIT[0]: sys.exit("Stopped by time limit.")


def add_default_arguments(parser):
    parser.add_argument('-s', metavar='style', dest='style',
                        type=str, default=None,
                        choices=['numpy', 'mptensor'],
                        help='set output style ("numpy" or "mptensor")')
    parser.add_argument('-t', metavar='time_limit', dest='time_limit',
                        type=float, default=None,
                        help='set time limit [sec]')
    parser.add_argument('-o', metavar='outfile', dest='outfile',
                        type=argparse.FileType('w'), default=sys.stdout,
                        help='write the result to outfile')
    parser.add_argument('infile',
                        type=argparse.FileType('r'),
                        help='tensor-network definition file')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code generator for tensor contruction")
    add_default_arguments(parser)
    args = parser.parse_args()

    main(args)
