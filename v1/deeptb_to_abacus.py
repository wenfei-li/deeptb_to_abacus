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

# Read basic info
with open('info.json') as info_file:
  info_dict = json.load(info_file)

nframe = -1
natom = -1
if(info_dict['nframes'] > 0):
    nframe = info_dict['nframes']
else:
    natoms = info_dict['natoms']

# Read atomic positions and lattice vectors, convert to Angstrom
an_to_bohr = 1.8897259886

positions = numpy.loadtxt('positions.dat').reshape(nframe,natom,3) # atomic positions
nframe,natom = positions.shape[:2]

latvecs = numpy.loadtxt('cell.dat').reshape(nframe,3,3) * an_to_bohr # convert to Bohr

if(info_dict['pos_type'].lower() == 'cart'): # convert to Cartesian in Bohr
    positions *= an_to_bohr
if(info_dict['pos_type'].lower() == 'direct'):
    positions = numpy.matmul(positions,latvecs)

# Read element types of atoms
# and extend the atomic positions with atomic index and numbers
atomic_numbers = numpy.loadtxt('atomic_numbers.dat').astype(int)[:natom]
elements = list(dict.fromkeys(atomic_numbers)) #preserve order

import glob
ps_name = []
orb_name = []
for el in elements:
    el_name = ELEMENT_NAMES[el-1]
    tmp = glob.glob(el_name+"*.upf") + glob.glob(el_name+"*.UPF")
    if(len(tmp)!=1):
        print("something wrong while finding pp file for element: " + el_name)
        print("pp file found to be: " + tmp)
    ps_name.append(tmp)

    tmp = glob.glob(el_name+"*.orb")
    if(len(tmp)!=1):
        print("something wrong while finding orb file for element: " + el_name)
        print("orb file found to be: " + tmp)
    orb_name.append(tmp)

# Check if atoms of the same type are adjacent, per requirement of ABACUS
assert checkContiguous(atomic_numbers,natom) == 1, 'atoms of the same type should be adjacent'

atomic_index = numpy.asarray([elements.index(i) for i in atomic_numbers]).reshape(natom,1)
atomic_numbers = atomic_numbers.reshape(natom,1)

positions_extend = numpy.zeros((nframe,natom,5)) # index,Zat,x,y,z
for iframe in range(nframe):
    positions_extend[iframe] = numpy.concatenate((atomic_index,atomic_numbers,positions[iframe]),axis=1)

# Read basis set info
basis_file = open('basis.dat','r')
basis_list_all = basis_file.read().splitlines()[:natom]

basis_list = []
for el in elements:
    ind = numpy.where(atomic_numbers == el)[0][0]
    basis_list.append(basis_list_all[ind])

rot_mat = []
for i in range(len(elements)):
    rot_mat.append(make_rot_mat(basis_list[i]))

basis_info = []
for i in range(len(elements)):
    basis_info.append(make_basis_info(basis_list[i]))

# Load data and save as npz format
f = h5py.File('DM.h5', 'r')
for i in range(nframe):
    dm_all = f[str(i)]
    fname_out = 'DM_'+str(i+1)+'.npz'

    dm_all_out = {}
    dm_all_out['lattice_vectors'] = latvecs[i]
    dm_all_out['atom_info'] = positions_extend[i]
    
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

# Generate input files for ABACUS
for i in range(nframe):
    stru_name = 'STRU'+str(i+1)

    with open(stru_name,'w') as stru_file:
        stru_file.write('LATTICE_CONSTANT\n')
        stru_file.write('1.0\n\n')
        
        stru_file.write('LATTICE_VECTORS\n')
        for i in range(3):
            stru_file.write(' '.join([str(j) for j in latvecs[iframe,i]])+'\n')
        stru_file.write('\n')

        stru_file.write('ATOMIC_SPECIES\n')
        for i in range(len(elements)):
            el_name = ELEMENT_NAMES[int(elements[i])-1]
            stru_file.write(' '.join([el_name,'1',ps_name[i][0]])+'\n')
        stru_file.write('\n')

        stru_file.write('NUMERICAL_ORBITAL\n')
        for i in range(len(elements)):
            el_name = ELEMENT_NAMES[int(elements[i])-1]
            stru_file.write(orb_name[i][0]+'\n')
        stru_file.write('\n')

        stru_file.write('ATOMIC_POSITIONS\n')
        stru_file.write('Cartesian\n\n')
        
        iat = 0
        for i in range(len(elements)):
            el_name = ELEMENT_NAMES[int(elements[i])-1]
            stru_file.write(el_name+'\n')
            stru_file.write('0.0\n')

            nat_type = numpy.sum(atomic_numbers == elements[i])
            stru_file.write(str(nat_type)+'\n')
            for j in range(nat_type):
                stru_file.write(' '.join([str(k) for k in positions_extend[iframe,iat,2:]])+'\n')
                iat += 1
            stru_file.write('\n')
