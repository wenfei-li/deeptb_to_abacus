import h5py
import numpy
import json

def make_rot_mat(basis): #example: 6s3p3d2f
    char_list = [*basis]
    nbasis_l = [int(char_list[i]) for i in range(len(char_list)) if i%2 == 0]
    nl = len(nbasis_l)

    l_list = []
    for il in range(nl):
        l_list.extend([il] * nbasis_l[il])

    lblock_siz = [2*l+1 for l in l_list]
    lblock_idx = [0]+list(numpy.cumsum(lblock_siz))
    nproj = lblock_idx[-1]

    rot_mat = numpy.zeros((nproj,nproj))
    for il in range(len(l_list)):
        l = l_list[il]
        if(l == 0):
            submat = numpy.eye(1)
        if(l == 1):
            submat = numpy.eye(3)[[1,2,0]]
            submat[[1,2]] *= -1
        if(l == 2):
            submat = numpy.eye(5)[[2,3,1,4,0]]
            submat[[1,2]] *= -1
        if(l == 3):
            submat = numpy.eye(7)[[3,4,2,5,1,6,0]]
            submat[[1,2,5,6]] *= -1
        if(l == 4):
            submat = numpy.eye(9)[[4,5,3,6,2,7,1,8,0]]
            submat[[1,2,5,6]] *= -1
        if(l == 5):
            submat = numpy.eye(11)[[5,6,4,7,3,8,2,9,1,10,0]]
            submat[[1,2,5,6,9,10]] *= -1

        rot_mat[lblock_idx[il]:lblock_idx[il+1],lblock_idx[il]:lblock_idx[il+1]] = submat
    return rot_mat

def make_basis_info(basis):
    char_list = [*basis]
    nbasis_l = [int(char_list[i]) for i in range(len(char_list)) if i%2 == 0]
    nl = len(nbasis_l)

    l_list = []
    for il in range(nl):
        l_list.extend([il] * nbasis_l[il])

    lblock_siz = [2*l+1 for l in l_list]
    lblock_idx = [0]+list(numpy.cumsum(lblock_siz))
    nproj = lblock_idx[-1]

    n = 0
    l_prev = l_list[0]
    basis_info = numpy.zeros((nproj,3))
    for il in range(len(l_list)):
        l = l_list[il]
        if(l != l_prev):
            n = 0
        if(l == 0):
            basis_info[lblock_idx[il]:lblock_idx[il+1],:] = numpy.array([[n,l,0]])
        if(l == 1):
            basis_info[lblock_idx[il]:lblock_idx[il+1],:] = numpy.array([[n,l,0],[n,l,1],[n,l,-1]])
        if(l == 2):
            basis_info[lblock_idx[il]:lblock_idx[il+1],:] = numpy.array([[n,l,0],[n,l,1],[n,l,-1],[n,l,2],[n,l,-2]])
        if(l == 3):
            basis_info[lblock_idx[il]:lblock_idx[il+1],:] = numpy.array([[n,l,0],[n,l,1],[n,l,-1],[n,l,2],[n,l,-2],[n,l,3],[n,l,-3]])
        if(l == 4):
            basis_info[lblock_idx[il]:lblock_idx[il+1],:] = numpy.array([[n,l,0],[n,l,1],[n,l,-1],[n,l,2],[n,l,-2],[n,l,3],[n,l,-3],[n,l,4],[n,l,-4]])
        if(l == 5):
            basis_info[lblock_idx[il]:lblock_idx[il+1],:] = numpy.array([[n,l,0],[n,l,1],[n,l,-1],[n,l,2],[n,l,-2],[n,l,3],[n,l,-3],[n,l,4],[n,l,-4],[n,l,5],[n,l,-5]])
        l_prev = l
        n += 1

    return basis_info

def checkContiguous(arr, n):
    # Keep track of visited elements
    visited = set()
    visited.add(arr[0])
 
    for i in range(1, n):
        if (arr[i] == arr[i - 1]):
            continue
        elif arr[i] in visited:
            return 0
        visited.add(arr[i])
    return 1

ELEMENT_NAMES = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
    'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
    'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
    'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
]

# Load STRU file
stru_file = open('STRU','r').readlines()
stru_file = [line for line in stru_file if line.strip()]

# split into blocks
keyword = ['ATOMIC_SPECIES','LATTICE_CONSTANT','LATTICE_VECTORS',
           'ATOMIC_POSITIONS','NUMERICAL_ORBITAL']

stru_blocks = {}
current_key = None
current_list = []
for line in stru_file:
    if any(kw == line.split()[0] for kw in keyword):
        if current_key:
            stru_blocks[current_key] = current_list
        current_key = line.strip()
        current_list = []
    else:
        current_list.append(line.strip())
stru_blocks[current_key] = current_list

# count number of elements and corresponding number of atoms
elements = [line.split()[0] for line in stru_blocks['ATOMIC_SPECIES']]
nat_per_type = []
for el in elements:
    i = stru_blocks['ATOMIC_POSITIONS'].index(el)
    nat_per_type.append(int(stru_blocks['ATOMIC_POSITIONS'][i+2]))
atomic_index = []
atomic_numbers = []
for i in range(len(elements)):
    atomic_index.extend([i]*nat_per_type[i])
    Zat = ELEMENT_NAMES.index(elements[i])+1
    atomic_numbers.extend([Zat]*nat_per_type[i])

atomic_index = numpy.asarray(atomic_index)
atomic_numbers = numpy.asarray(atomic_numbers)

# extract basis info
basis_list = []
basis_info = []
rot_mat = []
for i in range(len(elements)):
    el = elements[i]
    basis = stru_blocks['NUMERICAL_ORBITAL'][i].split('_')[4].replace('.orb','')
    basis_list.append(basis)
    basis_info.append(make_basis_info(basis))
    rot_mat.append(make_rot_mat(basis))

# convert latvec and atomic positions data
lat0 = float(stru_blocks['LATTICE_CONSTANT'][0].split()[0])

latvec = []
for i in range(3):
    for j in range(3):
        latvec.append(float(stru_blocks['LATTICE_VECTORS'][i].split()[j])*lat0)
latvec = numpy.asarray(latvec).reshape(3,3)

pos_type = stru_blocks['ATOMIC_POSITIONS'][0].split()[0].lower()
positions = []
ind = 4
for i in range(len(elements)):
    for j in range(nat_per_type[i]):
        position = numpy.asarray(stru_blocks['ATOMIC_POSITIONS'][ind].split()[:3]).astype(float)
        if(pos_type == 'cartesian' or pos_type == 'cart'):
            position = position * lat0
        elif(pos_type == 'direct'):
            position = numpy.matmul(position,latvec)
        else:
            raise ValueError('Unknown position type')

        positions.append(position)
        ind += 1
    ind += 3

positions = numpy.asarray(positions).astype(float).reshape(-1,3)
positions_extend = numpy.concatenate((atomic_index.reshape(-1,1),atomic_numbers.reshape(-1,1),positions),axis=1)
numpy.savetxt('positions_extend.dat',positions_extend)

# create output dictionary

dm_all_out = {}
dm_all_out['lattice_vectors'] = latvec
dm_all_out['atom_info'] = positions_extend

# Load data and save as npz format
f = h5py.File('DM.h5', 'r')
dm_all = f['0']
fname_out = 'output_DM0.npz'

for i in range(len(elements)):
    basis_name = 'orbital_info_'+str(i)
    dm_all_out[basis_name] = basis_info[i]

for submat in dm_all.keys():
    i,j,rx,ry,rz=submat.split('_')
    if(int(i)>int(j)):
        continue
    submat_name = '_'.join(['mat',i,j,rx,ry,rz])
    dm = numpy.asarray(dm_all[submat])
    dm_rot = numpy.einsum('ij,mi,nj->mn',dm,rot_mat[int(atomic_index[int(i)])],rot_mat[int(atomic_index[int(j)])])

    dm_all_out[submat_name] = dm_rot #+ numpy.random.rand(*dm_rot.shape)*1e-5
numpy.savez(fname_out,**dm_all_out)