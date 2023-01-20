import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np

#有四种基本的图的类型： Graph无向图， DiGraph有向图， MultiGraph多重无向图，MultiDiGraph多重有向图
import networkx as nx

#Chem负责基础常用的化学功能，比如读写分子，子结构搜索，分子美化等
from rdkit import Chem

#分子性质也被称为描述符，descriptors包含了大量的分子描述符的计算方法
from rdkit.Chem import Descriptors

#AllChem 负责高级但是不常用的化学功能，区分他们的目的是为了加速载入速度，同时也可以简化使用
from rdkit.Chem import AllChem

from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from torch.utils import data
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
from itertools import repeat, product, chain


# allowable node and edge features
#这个字典包含了节点和边
allowable_features = {
    #原子序号表，从1到199
    'possible_atomic_num_list' : list(range(1, 119)), 
    
    #形式电荷（英語：Formal charge，FC）是指在分子中，假定所有化学键的电子在原子之间均等共享（不考虑相对电负性），一个原子所分配到的电荷。
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], 

    #手性分子与其镜像不重合，像左手和右手一样
    'possible_chirality_list' : [
        
        # rdkit中的几种不同的手性分子
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        
        #Tetrahedral是四面体的意思
        #CW是clockwise
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        #CCW是counterclockwise
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    #可能的杂化轨道列表
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        
        #sp轨道由一个s一个p组成，linear，呈180度角
        Chem.rdchem.HybridizationType.SP,
        
        #sp2 是1s2p组成，trigonal planar，呈120度角
        Chem.rdchem.HybridizationType.SP2,
        
        #sp3， 1s3p，tetrahedral，呈109.5度角
        Chem.rdchem.HybridizationType.SP3, 
        
        #1s3p1d
        Chem.rdchem.HybridizationType.SP3D,
        
        #1s3p2d
        Chem.rdchem.HybridizationType.SP3D2, 
        
        #其他
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    
    #可能有多少个氢原子
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    
    #在open Babel的说法中，valence指一个原子有多少个键而不是化合价？这个不清楚测的是什么
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    
    #degree应该就是这个原子有多少个键吧？可能也不是？一个原子还能有十个键的吗？
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    
    #可能的键的类型
    'possible_bonds' : [
        #单键
        Chem.rdchem.BondType.SINGLE,
        #双键
        Chem.rdchem.BondType.DOUBLE,
        #三键
        Chem.rdchem.BondType.TRIPLE,
        #芳香键
        Chem.rdchem.BondType.AROMATIC
    ],
    
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        # end up right
        Chem.rdchem.BondDir.ENDUPRIGHT,
        # end down right
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

#看起来像是把分子转化成图表示，的simple方法
def mol_to_graph_data_obj_simple(mol):
    #将rdkit给出的分子表示转化为pytorch geometric包所要求的图表示
    #用了简化版的原子和键的特征，用目录表示
    #输入的是rdkit的分子object
    #输出， x:, edge_index:, edge_attr
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    #对于原子来说，有两个特征，一个是原子的类型，一个是原子的手性标签？为什么说原子有手性？不都是分子有手性吗？
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    #建一个特征的list
    atom_features_list = []
    #对于分子图中的每一个原子来说， 原来可以用GetAtoms()来取原子阿
    for atom in mol.GetAtoms():
        #原子的特征，用了前面提到的存有节点和键特征的字典
        #对于每个原子建一个原子array，将对应特征的index存在array里面用逗号隔开，所以每个原子都会生成一个二维的array eg:[2,1]
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
            'possible_chirality_list'].index(atom.GetChiralTag())]
        
        #然后将得出的所有原子的二维小feature放到前面建的list里面
        atom_features_list.append(atom_feature)
        
    #之后为原子的特征建一个tensor x， 里面是type为long的 [原子数, 2] 的array
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    #对于化学键来说也有两个特征，键的类型和键的方向
    num_bond_features = 2   # bond type, bond direction
    #如果一个分子中有键的话
    if len(mol.GetBonds()) > 0: # mol has bonds
        #建一个键的列表
        edges_list = []、
        #建一个键的特征的列表
        edge_features_list = []
        #遍历rdkit分子中的化学键
        for bond in mol.GetBonds():
            #分子中的键是有方向的
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            #获取可能的键的类型的index和代表化学键方向的index
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                                            'possible_bond_dirs'].index(
                bond.GetBondDir())]
            
            #以下几行难道是说这个键没有方向吗？为什么正序倒序都要存？直接存成无向图不就好了？反正键的特征是一样的？
            #将键的起止点正序存到键的列表中
            edges_list.append((i, j))
            #将键的特征存入键的特征列表
            edge_features_list.append(edge_feature)
            #将键的起止点倒序存入键的列表
            edges_list.append((j, i))
            #把键的特征往键的列表里再存一遍
            edge_features_list.append(edge_feature)
            
          
        #C00 format由三个array组成，第一个array存的是matrix中所有非零的值（从上到下，从左到右），第二个存的是这些值的row indexes，第三个array存的是这些值的column indexes
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        #edge list是[键数*2, 2]的array， T一下就是[2, 键数*2]的array,这个为什么能代表图的连通性connectivity呢，这不是只有一个array吗为什么说是COO format（Coordinate format）的呢？一定要用long来存吗？
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        
        #把[键数*2, 2]的键的特征列表建成tensor存到edge_attr中
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    #如果分子中不含有键
    else:   # mol has no bonds
        #会生成一个[[],[]]的tensor？那既然里面没有数的话为什么要调torch.empty？
        edge_index = torch.empty((2, 0), dtype=torch.long)
        #这个就是生成一个[]的tensor？
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    
    #最后将需要return的三个值用Data()封装一下，这个Data是torch.geometric.data中调的class，
    #Data(x: Optional[Tensor] = None, 
    #     edge_index: Optional[Tensor] = None, 
    #     edge_attr: Optional[Tensor] = None, 
    #     y: Optional[Tensor] = None, 
    #     pos: Optional[Tensor] = None, 
    #     **kwargs)
    #A data object describing a homogeneous graph. The data object can hold node-level, link-level and graph-level attributes. In general, Data tries to mimic the behaviour of a regular Python dictionary.
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    #最后将封装好的data return出去，就是适合torch.geometric用的图结构了
    return data

def graph_data_obj_to_mol_simple(data_x, data_edge_index, data_edge_attr):
    #将上一个function生成的可以用在torch geometric 的 data转化回rdkit的分子
    #用的是简化的分子和键的特征
    """
    Convert pytorch geometric data obj to rdkit mol object. NB: Uses simplified
    atom and bond features, and represent as indices.
    :param: data_x:
    :param: data_edge_index:
    :param: data_edge_attr
    :return:
    """
    #RWMol()用于分子的读写，在修改分子方面性能更好。提供一个可活动的分子并且共享mol对象的操作接口，修改完毕后用GetMol()获得最终的分子。
    mol = Chem.RWMol()

    
    # atoms
    # 把原子的特征tensor转化成numpy array 
    atom_features = data_x.cpu().numpy() 
    # 原子特征array的第一维的大小是总原子数 
    num_atoms = atom_features.shape[0]
    # 遍历特征array中的原子
    for i in range(num_atoms):
        #提出两个特征的index，原子序数的index和手性标签的index
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        
        #将原子序数具体的值用index和最开始建的字典提出来
        atomic_num = allowable_features['possible_atomic_num_list'][atomic_num_idx]
        
        #将手性标签的具体标签用index和最开始建的字典提出来
        chirality_tag = allowable_features['possible_chirality_list'][chirality_tag_idx]
        
        #输入原子序数在rdkit里创建一个原子
        atom = Chem.Atom(atomic_num)
        
        #将该原子的手性标签加回去
        atom.SetChiralTag(chirality_tag)
        
        #将这个原子加到可读写的rdkit分子中
        mol.AddAtom(atom)

    # bonds
    #再处理化学键
    #将两个有关边的tensor都转化回numpy
    edge_index = data_edge_index.cpu().numpy()  #[2, 键数*2]
    
    edge_attr = data_edge_attr.cpu().numpy()  #[键数*2, 2]
    
    #提取边的数量
    num_bonds = edge_index.shape[1]
    
    #因为当时正反向的同一条边都存了一次，所以loop的时候隔一个取一个
    for j in range(0, num_bonds, 2):
        
        # edge_index[0] 存着所有键的起点
        begin_idx = int(edge_index[0, j])
        # edge_index[1] 存着所有键的终点
        end_idx = int(edge_index[1, j])
        
        #从键的特征array中提取键类型和键方向的index
        bond_type_idx, bond_dir_idx = edge_attr[j]
        
        #用键类型的index去最前面建的字典里面提取键的类型
        bond_type = allowable_features['possible_bonds'][bond_type_idx]
        
        #用键的方向的index从最前面建的字典里面提取键的方向
        bond_dir = allowable_features['possible_bond_dirs'][bond_dir_idx]
        
        #将键和键的信息放到rdkit分子中
        mol.AddBond(begin_idx, end_idx, bond_type)
        
        # set bond direction
        #将刚才存进去的键提取出来
        new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
        #再把存好的键的方向加上
        new_bond.SetBondDir(bond_dir)

    # 用Chem.SanitizeMol(mol)可以将键的类型restore成芳香键
    # Chem.SanitizeMol(mol) # fails for COC1=CC2=C(NC(=N2)[S@@](=O)CC2=NC=C(
    # C)C(OC)=C2C)C=C1, when aromatic bond is possible
    # when we do not have aromatic bonds
    # Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)

    return mol

def graph_data_obj_to_nx_simple(data):
    #将为pytorch geometric准备的Data object转化成network x data object（这是个什么obj？）
    #这里面说nx obj是无向的所以可能会出现问题，但是nx obj其实也可以做成有向的呀
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    #创建一个无向图
    G = nx.Graph()

    # atoms
    #将原子的特征tensor转化为numpy array [原子数, 2]
    atom_features = data.x.cpu().numpy()
    #获得原子的数量
    num_atoms = atom_features.shape[0]
    #遍历原子
    for i in range(num_atoms):
        #获得原子序数的index和原子手性标签的index
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        #将这个存成无向图的一个节点， i是节点index，俩index直接存就行了？这么正好吗
        G.add_node(i, atom_num_idx=atomic_num_idx, chirality_tag_idx=chirality_tag_idx)
        #pass是占位符，不做任何事情
        pass

    # bonds
    #处理键
    #从tensor变成numpy array [2, 键数*2]
    edge_index = data.edge_index.cpu().numpy()
    #从tensor变成numpy array [键数*2, 2]
    edge_attr = data.edge_attr.cpu().numpy()
    #取键的数量
    num_bonds = edge_index.shape[1]
    #遍历不重复的键
    for j in range(0, num_bonds, 2):
        #获得键的起始原子的index
        begin_idx = int(edge_index[0, j])
        #获得键的结束原子的index
        end_idx = int(edge_index[1, j])
        #获得键的类型和方向的index
        bond_type_idx, bond_dir_idx = edge_attr[j]
        #如果nx图中的这两个原子之间没有边
        if not G.has_edge(begin_idx, end_idx):
            #那就创建一条边，将起始原子index， 终止原子index，键的类型，键的方向都输入进去
            G.add_edge(begin_idx, end_idx, bond_type_idx=bond_type_idx,
                       bond_dir_idx=bond_dir_idx)

    return G

def nx_to_graph_data_obj_simple(G):
    #将nx图转换回pytroch geometric所需要的Data obj
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # atoms
    #原子还是有两个特征，原子类型和手性标签
    num_atom_features = 2  # atom type,  chirality tag
    #创建原子的特征序列
    atom_features_list = []
    
    #遍历nx图中的节点，为什么要有一个_？
    for _, node in G.nodes(data=True):
        #将节点里存的原子序数的index和手性标签的index存成[2]的array
        atom_feature = [node['atom_num_idx'], node['chirality_tag_idx']]
        #再将小array存到特征序列里面去
        atom_features_list.append(atom_feature)
    #最后对原子特征序列建一个x的tensor [原子数，2]
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    #处理键，键也有两个特征，键的类型和键的方向
    num_bond_features = 2  # bond type, bond direction
    #如果nx图中有边的话
    if len(G.edges()) > 0:  # mol has bonds
        #建一个存键的list
        edges_list = []
        #建一个存键的特征的list
        edge_features_list = []
        #遍历nx图中的边 i是起始原子的index，j是终止原子的index，edge里面存着边的类型和方向
        for i, j, edge in G.edges(data=True):
            #将一条边的特征建成[2]的array[键类型index，键方向index]
            edge_feature = [edge['bond_type_idx'], edge['bond_dir_idx']]
            #将键正着存一遍到键list中
            edges_list.append((i, j))
            #将一条键的特征存到键特征list中
            edge_features_list.append(edge_feature)
            #反着存一遍
            edges_list.append((j, i))
            #键的特征也再存一遍
            edge_features_list.append(edge_feature)
            

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        #键的起止点们存成tensor，可能在geometic Data里面这个就会变成COO format吧，这里是看不出来为什么是COO
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        #键的特征也列表也存成tensor
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        #如果图中没有边的话存两个empty tensor
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    #最后将原子tensor x，键的起止点tensor， 键的特征tensor 输入geometric的Data()得到最终结果
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def get_gasteiger_partial_charges(mol, n_iter=12):
    #这个gasteiger partial charge是用两个阶段来分配电荷的算法，在第一阶段，给分子中的每个原子分配一个种子电荷。在第二阶段中，这些局部电荷开始沿着
    #与原子连接的键移动，移动方向取决于键两端所连接分子的电负性（electronegativities）。然后用relaxation算法进行迭代来调整电荷分配（默认迭代八次）
    #OpenEye不建议将这样计算出来的电荷模型用于分子间的相互左右。 Johann Gasteiger发明这个算法是用来比较不用分子结构下相关的官能团的相对reactivity
    """
    Calculates list of gasteiger partial charges for each atom in mol object.
    :param mol: rdkit mol object
    :param n_iter: number of iterations. Default 12 
    :return: list of computed partial charges for each atom.
    """
    
    #从rdkit里面调取写好的算法，输入有三个：1.rdkit分子，2.迭代次数，3.如果找不到原子的参数，是否报错，False的话未知原子的所有参数都设置为零，具有从迭代中删除该原子的效果
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    #对于rdkit分子中的每一个原子，获取这个原子的gasteigercharge，放到一个array里面
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    #就完事了
    return partial_charges

def create_standardized_mol_id(smiles):
    #这个方程用于生成标准化的rdkit分子id
    """
    :param smiles:
    :return: inchi
    """
    #如果这个SMILES可以生成rdkit分子的话
    if check_smiles_validity(smiles):
        # remove stereochemistry
        # 去除立体化学（有机化学的主要内容，研究有机物在三维空间内的结构与变化的化学分枝，由于化学键往往不是在二维平面上伸展的，于是就产生了相应的异构现象
        #AllChem负责高级但是不常用的化学功能，isomericSmiles True的话生成的rdkit分子就包含SMILES中存着的立体化学信息，Flase的话就去掉了
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        #然后再将去掉了立体化学信息的SMILES转化成rdkit分子
        mol = AllChem.MolFromSmiles(smiles)
        #如果转化成功的话
        if mol != None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            #如果有不同种类的SMILES生成的话，选最大的那个分子
            if '.' in smiles: # if multiple species, pick largest molecule
                # 先获取不同species的rdkit分子表
                mol_species_list = split_rdkit_mol_obj(mol)
                #选出那个拥有原子数最多的分子
                largest_mol = get_largest_mol(mol_species_list)
                #inchi = international chemical identifier 国际化学表示符
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            #return 后面还可以不写东西噢
            return
    else:
        return

#这是个dataset的类
class MoleculeDataset(InMemoryDataset):
    #初始化
    def __init__(self,
                 #这个dataset保存在哪个根目录下，应该包含一个raw文件夹（里面有含有SMILES的文件），一个processed文件夹（可以是空的也可以是上一次process完的文件）
                 root,
                 #data = None,
                 #slices = None,
                 #一个方程，输入geometric Data obj，输出一个transformed版本，每次access前这个data obj都会被transform
                 transform=None,
                 #同上，只是在存到disk之前被transform
                 pre_transform=None,
                 #输入一个geometric Data，返回一个boolean，表示了这个geometric Data是否应该留在最终的dataset中
                 pre_filter=None,
                 #数据集的名字
                 dataset='zinc250k',
                 #如果True的话不会加载Data obj
                 empty=False):
        #这个没有下载数据集的功能
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        
        self.dataset = dataset
        self.root = root
        #讲的不错的blog，可以跑一下试试
        #https://blog.csdn.net/wo198711203217/article/details/84097274
        #在类的继承中，如果重新定义某个方法，该方法会覆盖父类的同名方法，但有时我们希望能同时实现父类的功能，这个时候我们就需要调用父类的方法了
        #super最常见的一个用法就是在子类中调用父类初始化的方法
        #def super(cls, inst):
        #获取inst的MRO列表
        #    mro = inst.__class__.mro()
        #查找cls在当前MRO列表中的index，并返回他的下一个类
        #    return mro[mro.index(cls)+1]
        #cls 代表类，inst代表实例，
        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        #这些继承的是InMemoryDataset的值
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        #如果empty不是True的话
        if not empty:
            #self.processed_paths是在下面的def中定义的，这里还不知道讲了啥？
            self.data, self.slices = torch.load(self.processed_paths[0])

    #用来获取geometric data的def
    def get(self, idx):
        #先创建一个空的geometric Data obj
        data = Data()
        #对于data中的key来说，这个key存了什么？
        for key in self.data.keys:
            #从data里面按照key获取item，先看后面的####
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.cat_dim(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data

    #python内置的装饰器，可以讲一个方法变成属性来调用，后面不用加()，创建只读属性
    #由于python进行属性的定义时，没办法设置私有属性，因此要通过@property的方法来进行设置，这样可以隐藏属性名，让用户进行使用的时候无法随意修改
    #。 相当于用了一个调用def来封装了原本的属性名
    @property
    #原先的文件名
    def raw_file_names(self):
        #获取系统中的文件名列表
        file_name_list = os.listdir(self.raw_dir)
        #认为我们的文件下只有一个raw 文件
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    #定义一个文件名用来处理完的geometric data obj
    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'
    
    #这个方程就是专门用来说这个类不能从网上下载，要直接把数据放到目标文件夹中
    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')
    #
    def process(self):
        #创建一个存SMILES的列表
        data_smiles_list = []
        #创建一个存obj的列表？
        data_list = []
        
        #这个zinc_standard_agent代表了什么
        if self.dataset == 'zinc_standard_agent':
            #输入的路径就是raw文件夹下面的第一个文件（csv格式）
            input_path = self.raw_paths[0]
            #读取这个csv到dataframe里面去，这个csv 的sep = ‘/t’，数据类型是string
            #compression是可以读取压缩文件的，有{'infer','gzip','bz2','zip','xz',None}这几种，默认是infer
            input_df = pd.read_csv(input_path, sep=',', compression='gzip',
                                   dtype='str')
            #把读到的csv里面的‘smiles’单独存成list
            smiles_list = list(input_df['smiles'])
            #这种类型的csv里面还含有zinc_id，这个是这类dataset特定的嘛？
            zinc_id_list = list(input_df['zinc_id'])
            #遍历这个数据集
            for i in range(len(smiles_list)):
                #遍历的时候把到第几位了打印出来
                print(i)
                #s是一条数据中的SMILES，
                s = smiles_list[i]
                #每个sample中只含有原子数量最多的那个specie，还是没搞明白分子的species是什么
                # each example contains a single species
                
                #try是用来验证这段代码报不报错的
                #except是用来处理报错的
                #else当代码不报错的时候运行代码
                #finally是不管try和expect，运行代码
                try:
                    #生成rdkit分子
                    rdkit_mol = AllChem.MolFromSmiles(s)
                    #如果能生成的话
                    if rdkit_mol != None:  # ignore invalid mol objects
                        # # convert aromatic bonds to double bonds
                        # Chem.SanitizeMol(rdkit_mol,
                        #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                        #就将这个rdkit分子转化成geometric的Data obj
                        data = mol_to_graph_data_obj_simple(rdkit_mol)
                        # manually add mol id
                        #然后再手动添加上它的分子id，
                        #lstrip()用来截掉字符串左边的空格或者指定字符，这里是截掉开头的0
                        id = int(zinc_id_list[i].split('ZINC')[1].lstrip('0'))
                        #将id转化成tensor
                        data.id = torch.tensor(
                            [id])  # id here is zinc id value, stripped of
                        # leading zeros
                        #将geometric data加在data列表里
                        data_list.append(data)
                        #把相应的SMIELS加在smiles的列表里
                        data_smiles_list.append(smiles_list[i])
                except:
                    #如果上面那段不报错就继续执行
                    continue
        
        #如果dataset是属于另一种的话
        elif self.dataset == 'chembl_filtered':
            ### get downstream test molecules.
            #引入scaffold split
            from splitters import scaffold_split

            ### 
            #存了一堆数据集的文件夹名字，后面要遍历这些名字
            downstream_dir = [
            'dataset/bace',
            'dataset/bbbp',
            'dataset/clintox',
            'dataset/esol',
            'dataset/freesolv',
            'dataset/hiv',
            'dataset/lipophilicity',
            'dataset/muv',
            # 'dataset/pcba/processed/smiles.csv',
            'dataset/sider',
            'dataset/tox21',
            'dataset/toxcast'
            ]
            
            #建一个空set（）
            downstream_inchi_set = set()
            #遍历上面写的那一串路径
            for d_path in downstream_dir:
                print(d_path)
                #数据集的名字是/后面的那个所以是split[1]
                dataset_name = d_path.split('/')[1]
                #路径就是上面array里的某一个元素，dataset name是/后面跟着的东西，这里再创建一个MoleculeDataset obj
                downstream_dataset = MoleculeDataset(d_path, dataset=dataset_name)
                #下面用的SMILES是读取了处理之后的SMILES，使其变成一个list
                downstream_smiles = pd.read_csv(os.path.join(d_path,
                                                             'processed', 'smiles.csv'),
                                                header=None)[0].tolist()
                #在assert后面的判断为False的时候触发异常
                assert len(downstream_dataset) == len(downstream_smiles)
                
                #从deepchem的splitters里面引入的scaffold_split方程，
                #输入：1. 一个dataset，最低要保证些啥呢 2.SMILES的list，3.task的id？ 4. null_value设置成什么的意思吗
                #5. train valid test的比例，6.是否返回SMILES
                #输出有四个值？前三个不要，后面的是按比例分好的三个集的SMILES
                
                
                _, _, _, (train_smiles, valid_smiles, test_smiles) = scaffold_split(downstream_dataset, downstream_smiles, task_idx=None, null_value=0,
                                   frac_train=0.8,frac_valid=0.1, frac_test=0.1,
                                   return_smiles=True)

                ### remove both test and validation molecules
                #把test和valid的smiles存到一个里面后面用
                remove_smiles = test_smiles + valid_smiles
                #创建一个存id的list
                downstream_inchis = []
                #遍历上面的test和vaild的smiles
                for smiles in remove_smiles:
                    #分出不同的species
                    species_list = smiles.split('.')
                    #遍历所有的species并记录他们的inchi，而不只是原子数最多的那个分子的inchi
                    for s in species_list:  # record inchi for all species, not just
                     # largest (by default in create_standardized_mol_id if input has
                     # multiple species)
                        inchi = create_standardized_mol_id(s)
                        downstream_inchis.append(inchi)
                #update这个set就是把（）里面的值加到set里面去
                downstream_inchi_set.update(downstream_inchis)
            #这个是个啥写法，大概意思就是加载raw文件夹中的数据集了，返回SMILES列表，rdkit分子，啥？，和标签
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_chembl_with_labels_dataset(os.path.join(self.root, 'raw'))
            
            #打印一下开始处理了
            print('processing')
            #遍历取到的rdkit分子：
            for i in range(len(rdkit_mol_objs)):
                #打印一下进行到哪里了
                print(i)
                #单独取一个出来
                rdkit_mol = rdkit_mol_objs[i]
                #如果有rdkit分子值的话
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    #计算分子的摩尔质量
                    mw = Descriptors.MolWt(rdkit_mol)
                    #如果摩尔质量大于50小于900
                    if 50 <= mw <= 900:
                        #就正常的计算一个inchi
                        inchi = create_standardized_mol_id(smiles_list[i])
                        #如果出来结果了并且结果没有跟已有的inchi重复的话
                        if inchi != None and inchi not in downstream_inchi_set:
                            #就将这个rdkit分子转化成geometric Data obj
                            data = mol_to_graph_data_obj_simple(rdkit_mol)
                            # manually add mol id
                            #然后在将分子在数据集中的index当作id。转化成tensor
                            data.id = torch.tensor(
                                [i])  # id here is the index of the mol in
                            # the dataset
                            #将label加到geometric Data obj里面
                            #[行，列]，冒号表示取所有值，有数字的话代表第几位之后的，或者冒号两边有数字的话代表一个取值范围
                            #这里就是取了第i行所有的值，但是说实在的前面labels的标签只取了labels这一个column呀，难道做数据集的时候需要把所有label放到这一个column里面吗
                            data.y = torch.tensor(labels[i, :])
                            # fold information
                            #这个折信息又是什么
                            #如果i属于第一个fold的话
                            if i in folds[0]:
                                #geometric Data obj里面的tensor就是0
                                data.fold = torch.tensor([0])
                            elif i in folds[1]:
                                #在1tensor就存1
                                data.fold = torch.tensor([1])
                            else:
                                #其他的tensor就存成2
                                data.fold = torch.tensor([2])
                            #然后把处理好的data放到data列表里面去
                            data_list.append(data)
                            #处理好的SMILES也放到列里面
                            data_smiles_list.append(smiles_list[i])
        #下面是对于单个数据集的处理方式
        #对于tox21来说
        elif self.dataset == 'tox21':
            #加载路径raw文件下的tox数据们
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            #遍历所有的SMILES和分子
            for i in range(len(smiles_list)):
                print(i)
                #取出一个rdkit分子
                rdkit_mol = rdkit_mol_objs[i]
                ## convert aromatic bonds to double bonds
                #Chem.SanitizeMol(rdkit_mol,
                                 #sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                #然后把rdkit分子转化成geometric Data obj
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                #加上这个分子在数据集里的index当作这个sample的id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                #把label加上
                data.y = torch.tensor(labels[i, :])
                #存起来
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'hiv':
            #加载数据们
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(self.raw_paths[0])
            #遍历数据集
            for i in range(len(smiles_list)):
                print(i)
                
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                #rdkit分子 转成geometric图
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                #加id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                #取label，这个label就只有一维，是几个task就有几维吗？
                data.y = torch.tensor([labels[i]])
                #存数据
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'bace':
            #读数据
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_bace_dataset(self.raw_paths[0])
            #遍历
            for i in range(len(smiles_list)):
                print(i)
               
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                #转成geometric data
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                #加id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset               
                #取label，fold信息，这个fold到底是个啥吗
                data.y = torch.tensor([labels[i]])
                data.fold = torch.tensor([folds[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = \
                _load_bbbp_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    #就一个task，没有fold信息
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = \
                _load_clintox_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    #好多个task
                    data.y = torch.tensor(labels[i, :])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = \
                _load_esol_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                #一个task
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_freesolv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                #一个task
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = \
                _load_lipophilicity_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                #一个task
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'muv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_muv_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                #多个task
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'pcba':
            smiles_list, rdkit_mol_objs, labels = \
                _load_pcba_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                #多task
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
        
        #这个数据集又不一样了
        elif self.dataset == 'pcba_pretrain':
            #提取数据
            smiles_list, rdkit_mol_objs, labels = \
                _load_pcba_dataset(self.raw_paths[0])
            #提取现有的inchi
            downstream_inchi = set(pd.read_csv(os.path.join(self.root,
                                                            'downstream_mol_inchi_may_24_2019'),
                                               sep=',', header=None)[0])
            #遍历
            for i in range(len(smiles_list)):
                print(i)
                #如果有多个species就不考虑
                if '.' not in smiles_list[i]:   # remove examples with
                    # multiples species
                    
                    rdkit_mol = rdkit_mol_objs[i]
                    #再测分子的摩尔质量，太大太小的都去掉
                    mw = Descriptors.MolWt(rdkit_mol)
                    if 50 <= mw <= 900:
                        #用smiles自己生成一个inchi
                        inchi = create_standardized_mol_id(smiles_list[i])
                        #如果可以生成且跟前面的不重复
                        if inchi != None and inchi not in downstream_inchi:
                            # # convert aromatic bonds to double bonds
                            # Chem.SanitizeMol(rdkit_mol,
                            #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                            #变成geometric Data
                            data = mol_to_graph_data_obj_simple(rdkit_mol)
                            # manually add mol id
                            #加id
                            data.id = torch.tensor(
                                [i])  # id here is the index of the mol in
                            # the dataset
                            #多task的label
                            data.y = torch.tensor(labels[i, :])
                            #存
                            data_list.append(data)
                            data_smiles_list.append(smiles_list[i])

        # elif self.dataset == ''

        elif self.dataset == 'sider':
            #加载数据
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            #遍历
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # # convert aromatic bonds to double bonds
                # Chem.SanitizeMol(rdkit_mol,
                #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                #多task
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'toxcast':
            #加载
            smiles_list, rdkit_mol_objs, labels = \
                _load_toxcast_dataset(self.raw_paths[0])
            #遍历
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    #多task
                    data.y = torch.tensor(labels[i, :])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'ptc_mr':
            #这个是在tox21地下的一个任务吗
            
            input_path = self.raw_paths[0]
            #读进来，readcsv 输入中的names=[]可定义columns
            input_df = pd.read_csv(input_path, sep=',', header=None, names=['id', 'label', 'smiles'])
            smiles_list = input_df['smiles']
            #dataframe.values返回dataframe的numpy表达形式
            labels = input_df['label'].values
            #遍历
            for i in range(len(smiles_list)):
                print(i)
                #提取一个SMILES
                s = smiles_list[i]
                #转化成rdkit分子
                rdkit_mol = AllChem.MolFromSmiles(s)
                #如果转化成功了
                if rdkit_mol != None:  # ignore invalid mol objects
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    #再转化成geometric Data
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    #加id
                    data.id = torch.tensor(
                        [i])
                    #一个task的label
                    data.y = torch.tensor([labels[i]])
                    #存
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])

        elif self.dataset == 'mutag':
            #加载SMILES路径
            smiles_path = os.path.join(self.root, 'raw', 'mutag_188_data.can')
            # smiles_path = 'dataset/mutag/raw/mutag_188_data.can'
            #加载label路径
            labels_path = os.path.join(self.root, 'raw', 'mutag_188_target.txt')
            # labels_path = 'dataset/mutag/raw/mutag_188_target.txt'
            #取得SMILES列表和label的列表
            smiles_list = pd.read_csv(smiles_path, sep=' ', header=None)[0]
            labels = pd.read_csv(labels_path, header=None)[0].values
            #遍历
            for i in range(len(smiles_list)):
                print(i)
                #SMILES成功转rdkit分子之后再转geometric Data
                s = smiles_list[i]
                rdkit_mol = AllChem.MolFromSmiles(s)
                if rdkit_mol != None:  # ignore invalid mol objects
                    # # convert aromatic bonds to double bonds
                    # Chem.SanitizeMol(rdkit_mol,
                    #                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                    data = mol_to_graph_data_obj_simple(rdkit_mol)
                    # manually add mol id
                    #加id
                    data.id = torch.tensor(
                        [i])
                    #一个task
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
                    

        else:
            #如果没有上面提到的数据集名字就raise Err
            raise ValueError('Invalid dataset name')
        #如果这个filter有东西的话
        if self.pre_filter is not None:
            #将data list里面的所有geometric Data obj过一遍filter（）True留下False去掉
            data_list = [data for data in data_list if self.pre_filter(data)]
        #如果transform有东西的话
        if self.pre_transform is not None:
            #就把list里面所有data过一遍transform（）
            data_list = [self.pre_transform(data) for data in data_list]
        #将处理之后的SMILES列表变成Series，用csv存起来
        #Series是带标签的一维数组，其中的标签也可以叫做index
        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)
        #collate就是pytorch的DataLoader需要传递的参数，把batchge
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# NB: only properly tested when dataset_1 is chembl_with_labels and dataset_2
# is pcba_pretrain
#
def merge_dataset_objs(dataset_1, dataset_2):
    #用来合并两个数据集中的obj，不管这个分子的identity，假定两个数据集都含有多个label，最终的label dimension会是两个数据集的dimension加起来那么大
    #y1 1310 + y2 128
    #0 1310 + y2 128
    # y1 1310 + 0 128
    """
    Naively merge 2 molecule dataset objects, and ignore identities of
    molecules. Assumes both datasets have multiple y labels, and will pad
    accordingly. ie if dataset_1 has obj_1 with y dim 1310 and dataset_2 has
    obj_2 with y dim 128, then the resulting obj_1 and obj_2 will have dim
    1438, where obj_1 have the last 128 cols with 0, and obj_2 have
    the first 1310 cols with 0.
    :return: pytorch geometric dataset obj, with the x, edge_attr, edge_index,
    new y attributes only
    """
    #先算出两个数据集的y的维度
    d_1_y_dim = dataset_1[0].y.size()[0]
    d_2_y_dim = dataset_2[0].y.size()[0]
    #建一个空data列表
    data_list = []
    # keep only x, edge_attr, edge_index, padded_y then append
    #遍历第一个数据集
    for d in dataset_1:
        #原先的label先存一下
        old_y = d.y
        #新的是老的后面接上len(y2)个零
        new_y = torch.cat([old_y, torch.zeros(d_2_y_dim, dtype=torch.long)])
        #再将改好y的data放回去
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))
    #遍历第二个数据集
    for d in dataset_2:
        #存旧y
        old_y = d.y
        #新y再前面加上len(y1)个零
        new_y = torch.cat([torch.zeros(d_1_y_dim, dtype=torch.long), old_y.long()])
        #将改好的data存回去
        data_list.append(Data(x=d.x, edge_index=d.edge_index,
                              edge_attr=d.edge_attr, y=new_y))
    #新建一个dataset，随便选一个路径和数据集
    # create 'empty' dataset obj. Just randomly pick a dataset and root path
    # that has already been processed
    new_dataset = MoleculeDataset(root='dataset/chembl_with_labels',
                                  dataset='chembl_with_labels', empty=True)
    # collate manually
    
    new_dataset.data, new_dataset.slices = new_dataset.collate(data_list)

    return new_dataset

#生成环形指纹
def create_circular_fingerprint(mol, radius, size, chirality):
    #输入rdkit分子，半径，大小？，手性，为什么手性这么重要呀？
    """

    :param mol:
    :param radius:
    :param size:
    :param chirality:
    :return: np array of morgan fingerprint
    """
    #相当于把另一个方程封装了一下并且把结果转成了numpy arr
    #这个方程是rdkit.chem里面的，
    #https://blog.csdn.net/dreadlesss/article/details/105976113
    #morgan fingerprints摩根分子指纹是一种圆形指纹，也属于拓扑型指纹，是通过对标准的摩根算法进行改造后得到
    #可以大致等同于扩展连通性指纹（ECFP）
    #计算速度快，没有预定义，可以表示无穷多种不同的分子特征，可以包含手性信息，每个元素代表一种特定的子结构。
    #这类指纹设计的最初目的是用于搜索与活性相关的分子特征，而不是子结构搜索。
    #步骤：1.原子初始化，为每个重原子分配一个整数标识符 2. 迭代更新，以每个重原子为中心，将周围一圈的原子合并起来，直到达到指定半径
    #3. 特征生成，对子结构进行运算，并生成特征列表
    #nBits是长度
    #Draw.DrawMorganBit(m1,211414882, info)可以画出提取出的子结构的分子图
    
    fp = GetMorganFingerprintAsBitVect(mol, radius,
                                       nBits=size, useChirality=chirality)
    return np.array(fp)

#一个新的dataset，记录分子指纹的，data是调的torch.utils.data

class MoleculeFingerprintDataset(data.Dataset):
    def __init__(self, root, dataset, radius, size, chirality=True):
        #存的是字典的列表，每一个字典里都包含分子的环形指纹，标签，ID，和可能有的fold信息，这个fold是train/valid/test训练集的信息
        #输入：1. root：数据集所在的根目录，raw文件夹下面需要有含有SMILES的数据集，processed文件夹下面可以是空的也可以是之前处理好的
        #2. dataset是数据集的名字，又像上一个MoleculeDataset一样了，给每一个dataset都整个不同的出来
        #3. radius，环形分子的半径。4. size：分子指纹向量的长度。 5.chirality如果是true的话那就包含指纹的信息
        
        
        """
        Create dataset object containing list of dicts, where each dict
        contains the circular fingerprint of the molecule, label, id,
        and possibly precomputed fold information
        :param root: directory of the dataset, containing a raw and
        processed_fp dir. The raw dir should contain the file containing the
        smiles, and the processed_fp dir can either be empty or a
        previously processed file
        :param dataset: name of dataset. Currently only implemented for
        tox21, hiv, chembl_with_labels
        :param radius: radius of the circular fingerprints
        :param size: size of the folded fingerprint vector
        :param chirality: if True, fingerprint includes chirality information
        """
        self.dataset = dataset
        self.root = root
        self.radius = radius
        self.size = size
        self.chirality = chirality
        #这个load()，两个下划线__首先表示的是内置函数，
        #__双下划线也可以限制访问，self.__name = name,这样从类的外面就不能访问__name
        #在python中以双下划线开头和结尾的变量是特殊变量，特殊变量是可以直接访问的。
        #如果变量名前面只有一个下划线，表示此变量不要随便访问，虽然它可以被直接访问
        self._load()

    def _process(self):
        #建一个SMIELS的列表
        data_smiles_list = []
        #建一个data的列表
        data_list = []
        #如果dataset是这个的话，上一个MoleculeDataset里面也有这个数据集
        if self.dataset == 'chembl_with_labels':
            #加载SMILES， rdkit分子，fold，标签从raw里面加载出来
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_chembl_with_labels_dataset(os.path.join(self.root, 'raw'))
            print('processing')
            #遍历这个数据集
            for i in range(len(rdkit_mol_objs)):
                #进行到哪个分子了
                print(i)
                #rdkit分子拿一个出来
                rdkit_mol = rdkit_mol_objs[i]
                #如果有分子的话
                if rdkit_mol != None:
                    # 将芳香键转化成双键
                    # # convert aromatic bonds to double bonds
                    #这不是在生成环形指纹吗
                    fp_arr = create_circular_fingerprint(rdkit_mol,
                                                         self.radius,
                                                         self.size, self.chirality)
                    #把生成的numpy指纹转化成tensor
                    fp_arr = torch.tensor(fp_arr)
                    # manually add mol id
                    #手动加上分子在数据集中的index作为分子的id
                    id = torch.tensor([i])  # id here is the index of the mol in
                    #提取所有的label
                    y = torch.tensor(labels[i, :])
                    #在这里给分子分出来train vaild test集
                    # fold information
                    if i in folds[0]:
                        fold = torch.tensor([0])
                    elif i in folds[1]:
                        fold = torch.tensor([1])
                    else:
                        fold = torch.tensor([2])
                    #然后在data list里面接上含有分子指纹，ID，标签，fold的字典
                    data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y,
                                      'fold': fold})
                    #将SMIELS存到SMILES列表里面去
                    data_smiles_list.append(smiles_list[i])
        #tox21数据集
        elif self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(os.path.join(self.root, 'raw/tox21.csv'))
            print('processing')
            #遍历数据集
            for i in range(len(smiles_list)):
                print(i)
                # 取一个分子出来
                rdkit_mol = rdkit_mol_objs[i]
                ## convert aromatic bonds to double bonds
                #算它的环形指纹
                fp_arr = create_circular_fingerprint(rdkit_mol,
                                                        self.radius,
                                                        self.size,
                                                        self.chirality)
                #把算出来的指纹变成tensor
                fp_arr = torch.tensor(fp_arr)

                # manually add mol id
                # 手动加上ID，就是分子在数据集里面的index
                id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset#
                # 取所有的label并且转化成tensor
                y = torch.tensor(labels[i, :])
                # 这个tox21就没有fold信息可以存
                data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y})
                # 将SMILES存起来
                data_smiles_list.append(smiles_list[i])
                
        #hiv数据集
        elif self.dataset == 'hiv':
            #将SMILES，rdkit分子和label提出来
            smiles_list, rdkit_mol_objs, labels = \
                _load_hiv_dataset(os.path.join(self.root, 'raw/HIV.csv'))
            print('processing')
            #遍历
            for i in range(len(smiles_list)):
                print(i)
                #取一个rdkit分子
                rdkit_mol = rdkit_mol_objs[i]
                #创造环形指纹，这个跟芳香键还是双键有啥关系呢
                # # convert aromatic bonds to double bonds
                fp_arr = create_circular_fingerprint(rdkit_mol,
                                                        self.radius,
                                                        self.size,
                                                        self.chirality)
                #把得到的环形指纹转化成tensor
                fp_arr = torch.tensor(fp_arr)

                # manually add mol id
                # 手动加id
                id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                #存label
                y = torch.tensor([labels[i]])
                #存指纹
                data_list.append({'fp_arr': fp_arr, 'id': id, 'y': y})
                #存SMILES
                data_smiles_list.append(smiles_list[i])
        else:
            #如果不在以上的dataset里面就报错
            raise ValueError('Invalid dataset name')

        # save processed data objects and smiles
        #把目标文件夹的路径搞出来
        processed_dir = os.path.join(self.root, 'processed_fp')
        #把SMIELS转成series再存成csv
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(processed_dir, 'smiles.csv'),
                                  index=False,
                                  header=False)
        with open(os.path.join(processed_dir,
                                    'fingerprint_data_processed.pkl'),
                  'wb') as f:
            pickle.dump(data_list, f)

    #这个load就是先把路径给它提取出来
    def _load(self):
        processed_dir = os.path.join(self.root, 'processed_fp')
        # check if saved file exist. If so, then load from save
        #然后提出来所有文件名字的列表
        file_name_list = os.listdir(processed_dir)
        #如果这个处理之后的文件在这个路径下的话，就加载已经处理好的数据
        if 'fingerprint_data_processed.pkl' in file_name_list:
            with open(os.path.join(processed_dir,
                                   'fingerprint_data_processed.pkl'),
                      'rb') as f:
                self.data_list = pickle.load(f)
        # if no saved file exist, then perform processing steps, save then
        # reload
        else:
            #如果没有的话就process处理一个出来，存起来，再load
            self._process()
            self._load()
    #内置函数，计算这个dataset的大小
    def __len__(self):
        return len(self.data_list)
    #内置函数，取其中一个object
    def __getitem__(self, index):
        ## if iterable class is passed, return dataset objection
        #hasattr是python的内置函数，用于判断对象是否包含对应的属性，如果有该属性就True没有的话就False，就是index这个变量里面有没有“__iter__"
        if hasattr(index, "__iter__"):
            #如果有的话，那就在建一个新的dataset，然后把data放进去
            dataset = MoleculeFingerprintDataset(self.root, self.dataset, self.radius, self.size, chirality=self.chirality)
            dataset.data_list = [self.data_list[i] for i in index]
            #返回这个
            return dataset
        else:
            #如果没有的话就直接返回data的列表
            return self.data_list[index]

#又是一个轻易不要调取的内置函数
def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    #读取csv
    input_df = pd.read_csv(input_path, sep=',')
    #取SMILES
    smiles_list = input_df['smiles']
    #生成rdkit分子
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #所有task的列表
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
       'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    #一下子取所有的task的label
    labels = input_df[tasks]
    # convert 0 to -1
    #将0转成-1
    labels = labels.replace(0, -1)
    # convert nan to 0
    #将nan转成0
    labels = labels.fillna(0)
    #检查数据的长度有没有问题
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    #没问题就返回
    return smiles_list, rdkit_mol_objs_list, labels.values

#加载hiv数据集
def _load_hiv_dataset(input_path):
    """
    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    #读csv
    input_df = pd.read_csv(input_path, sep=',')
    #取SMILES
    smiles_list = input_df['smiles']
    #转rdkit分子
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #一个task的话就直接取label
    labels = input_df['HIV_active']
    # convert 0 to -1
    #0转-1
    labels = labels.replace(0, -1)
    # there are no nans
    #检查数据的可靠性
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    #没问题就返回
    return smiles_list, rdkit_mol_objs_list, labels.values

#加载bace数据集
def _load_bace_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array
    containing indices for each of the 3 folds, np.array containing the
    labels
    """
    #读csv
    input_df = pd.read_csv(input_path, sep=',')
    #读SMILES
    smiles_list = input_df['mol']
    #转rdkit分子
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #取label
    labels = input_df['Class']
    # convert 0 to -1
    #0转-1
    labels = labels.replace(0, -1)
    # there are no nans
    #取fold信息，fold就是自带的这个分子应该属于train valid还是test集
    folds = input_df['Model']
    folds = folds.replace('Train', 0)   # 0 -> train
    folds = folds.replace('Valid', 1)   # 1 -> valid
    folds = folds.replace('Test', 2)    # 2 -> test
    #检查数据的可靠性
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)
    #没问题就返回
    return smiles_list, rdkit_mol_objs_list, folds.values, labels.values

#加载bbbp数据集
def _load_bbbp_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    #读csv
    input_df = pd.read_csv(input_path, sep=',')
    #读SMILES
    smiles_list = input_df['smiles']
    #转rdkit分子
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #这里处理了一堆没有成功转成rdkit分子的SMLIES，但是我不明白为什么要把这个要从None转成None，
    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                                          rdkit_mol_objs_list]
    #如果rdkit分子是None的话，就把对应的SMILES也转成None
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    #取label
    labels = input_df['p_np']
    # convert 0 to -1
    #0转成-1
    labels = labels.replace(0, -1)
    # there are no nans
    #检查数据是否缺
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    #没缺的话就返回值，这里面的\是个啥意思
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values

#加载clintox数据集
def _load_clintox_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    #读csv
    input_df = pd.read_csv(input_path, sep=',')
    #读SMILES
    smiles_list = input_df['smiles']
    #转人dkit分子
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #如果没成功转成rdkit分子的话，就把None换成None，真的是没道理？不做会怎么样
    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    #如果rdkit分子None的话，SMILES也设成None
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    #task列表
    tasks = ['FDA_APPROVED', 'CT_TOX']
    #提取label
    labels = input_df[tasks]
    # convert 0 to -1
    #将0转成-1
    labels = labels.replace(0, -1)
    # there are no nans
    #检查数据的合理性
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    #没问题的话就返回
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values
# input_path = 'dataset/clintox/raw/clintox.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_clintox_dataset(input_path)

#加载esol数据集
def _load_esol_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    # NB: some examples have multiple species
    #读csv
    input_df = pd.read_csv(input_path, sep=',')
    #读SMILES
    smiles_list = input_df['smiles']
    #转rdkit分子
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #取label
    labels = input_df['measured log solubility in mols per litre']
    #检查数据
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    #没问题就返回
    return smiles_list, rdkit_mol_objs_list, labels.values
# input_path = 'dataset/esol/raw/delaney-processed.csv'
# smiles_list, rdkit_mol_objs_list, labels = _load_esol_dataset(input_path)

#加载freesolv数据集
def _load_freesolv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    #读取csv
    input_df = pd.read_csv(input_path, sep=',')
    #读SMILES
    smiles_list = input_df['smiles']
    #转rdkit分子
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #取label
    labels = input_df['expt']
    #检查数据
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    #没问题返回
    return smiles_list, rdkit_mol_objs_list, labels.values

#加载lipophilicity数据集
def _load_lipophilicity_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    #读csv
    input_df = pd.read_csv(input_path, sep=',')
    #读SMILES
    smiles_list = input_df['smiles']
    #转rdkit分子
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #取label
    labels = input_df['exp']
    #检查数据
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    #没问题就返回
    return smiles_list, rdkit_mol_objs_list, labels.values

#加载muv数据集
def _load_muv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    #读取csv
    input_df = pd.read_csv(input_path, sep=',')
    #取SMILES
    smiles_list = input_df['smiles']
    #转rdkit分子
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #取task列表
    tasks = ['MUV-466', 'MUV-548', 'MUV-600', 'MUV-644', 'MUV-652', 'MUV-689',
       'MUV-692', 'MUV-712', 'MUV-713', 'MUV-733', 'MUV-737', 'MUV-810',
       'MUV-832', 'MUV-846', 'MUV-852', 'MUV-858', 'MUV-859']
    #取所有task的label
    labels = input_df[tasks]
    # convert 0 to -1
    #0转-1
    labels = labels.replace(0, -1)
    # convert nan to 0
    #nan转0
    labels = labels.fillna(0)
    #检查数据
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    #没问题就返回
    return smiles_list, rdkit_mol_objs_list, labels.values

#加载sider数据集
def _load_sider_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    #读csv
    input_df = pd.read_csv(input_path, sep=',')
    #读SMILES
    smiles_list = input_df['smiles']
    #转rdkit分子
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    #列出task列表，这个列表也太长了吧
    tasks = ['Hepatobiliary disorders',
       'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
       'Investigations', 'Musculoskeletal and connective tissue disorders',
       'Gastrointestinal disorders', 'Social circumstances',
       'Immune system disorders', 'Reproductive system and breast disorders',
       'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
       'General disorders and administration site conditions',
       'Endocrine disorders', 'Surgical and medical procedures',
       'Vascular disorders', 'Blood and lymphatic system disorders',
       'Skin and subcutaneous tissue disorders',
       'Congenital, familial and genetic disorders',
       'Infections and infestations',
       'Respiratory, thoracic and mediastinal disorders',
       'Psychiatric disorders', 'Renal and urinary disorders',
       'Pregnancy, puerperium and perinatal conditions',
       'Ear and labyrinth disorders', 'Cardiac disorders',
       'Nervous system disorders',
       'Injury, poisoning and procedural complications']
    #取所有task的label
    labels = input_df[tasks]
    # convert 0 to -1
    #0转成-1
    labels = labels.replace(0, -1)
    #检查数据
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    #没问题就返回
    return smiles_list, rdkit_mol_objs_list, labels.value

#加载toxcast数据集
def _load_toxcast_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    # NB: some examples have multiple species, some example smiles are invalid
    #这个数据集涉及不同的分子species了，但是我还是不明白啥叫不同的species
    #读csv
    input_df = pd.read_csv(input_path, sep=',')
    #取SMILES
    smiles_list = input_df['smiles']
    #转rdkit分子
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    # Some smiles could not be successfully converted
    # to rdkit mol object so them to None
    # 处理了没成功转成rdkit分子的SMILES，将他们设置成None，可是他们本来就是None了啊？
    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    #没转成功的把SMILES也设制成None
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    # 直接除了第一列其他的都是task，转成list
    tasks = list(input_df.columns)[1:]
    #取label
    labels = input_df[tasks]
    # convert 0 to -1
    #0转化成-1
    labels = labels.replace(0, -1)
    # convert nan to 0
    #nan转化成0
    labels = labels.fillna(0)
    #检查数据
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    #没问题就返回，\的意思？
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values
#加载chembl数据集，这个长啊
def _load_chembl_with_labels_dataset(root_path):
    """
    Data from 'Large-scale comparison of machine learning methods for drug target prediction on ChEMBL'
    :param root_path: path to the folder containing the reduced chembl dataset
    :return: list of smiles, preprocessed rdkit mol obj list, list of np.array
    containing indices for each of the 3 folds, np.array containing the labels
    """
    # adapted from https://github.com/ml-jku/lsc/blob/master/pythonCode/lstm/loadData.py
    # first need to download the files and unzip:
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced.zip
    # unzip and rename to chembl_with_labels
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20Smiles.pckl
    # into the dataPythonReduced directory
    # wget http://bioinf.jku.at/research/lsc/chembl20/dataPythonReduced/chembl20LSTM.pckl

    # 1. load folds and labels
    f=open(os.path.join(root_path, 'folds0.pckl'), 'rb')
    folds=pickle.load(f)
    f.close()

    f=open(os.path.join(root_path, 'labelsHard.pckl'), 'rb')
    targetMat=pickle.load(f)
    sampleAnnInd=pickle.load(f)
    targetAnnInd=pickle.load(f)
    f.close()

    targetMat=targetMat
    targetMat=targetMat.copy().tocsr()
    targetMat.sort_indices()
    targetAnnInd=targetAnnInd
    targetAnnInd=targetAnnInd-targetAnnInd.min()

    folds=[np.intersect1d(fold, sampleAnnInd.index.values).tolist() for fold in folds]
    targetMatTransposed=targetMat[sampleAnnInd[list(chain(*folds))]].T.tocsr()
    targetMatTransposed.sort_indices()
    # # num positive examples in each of the 1310 targets
    trainPosOverall=np.array([np.sum(targetMatTransposed[x].data > 0.5) for x in range(targetMatTransposed.shape[0])])
    # # num negative examples in each of the 1310 targets
    trainNegOverall=np.array([np.sum(targetMatTransposed[x].data < -0.5) for x in range(targetMatTransposed.shape[0])])
    # dense array containing the labels for the 456331 molecules and 1310 targets
    denseOutputData=targetMat.A # possible values are {-1, 0, 1}

    # 2. load structures
    f=open(os.path.join(root_path, 'chembl20LSTM.pckl'), 'rb')
    rdkitArr=pickle.load(f)
    f.close()

    assert len(rdkitArr) == denseOutputData.shape[0]
    assert len(rdkitArr) == len(folds[0]) + len(folds[1]) + len(folds[2])

    preprocessed_rdkitArr = []
    print('preprocessing')
    for i in range(len(rdkitArr)):
        print(i)
        m = rdkitArr[i]
        if m == None:
            preprocessed_rdkitArr.append(None)
        else:
            mol_species_list = split_rdkit_mol_obj(m)
            if len(mol_species_list) == 0:
                preprocessed_rdkitArr.append(None)
            else:
                largest_mol = get_largest_mol(mol_species_list)
                if len(largest_mol.GetAtoms()) <= 2:
                    preprocessed_rdkitArr.append(None)
                else:
                    preprocessed_rdkitArr.append(largest_mol)

    assert len(preprocessed_rdkitArr) == denseOutputData.shape[0]

    smiles_list = [AllChem.MolToSmiles(m) if m != None else None for m in
                   preprocessed_rdkitArr]   # bc some empty mol in the
    # rdkitArr zzz...

    assert len(preprocessed_rdkitArr) == len(smiles_list)

    return smiles_list, preprocessed_rdkitArr, folds, denseOutputData
# root_path = 'dataset/chembl_with_labels'

def check_smiles_validity(smiles):
    #用于检测SMILES有效性的，能生成rdkit分子的就是True不能就是False
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False

def split_rdkit_mol_obj(mol):
    #具体来讲什么是分子的species？这个方程就是将那个结果给他分割开形成一个array （原先 species1.species2.species3 变成 [species1, species2, species3]) 
    #是SMILES会有不同的species用.隔开，不是mol里面，注释有问题
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    #先从rdkit分子生成一个包含立体结构信息的SMILES
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    #再将SMILES里面的不同species分出来
    smiles_list = smiles.split('.')
    mol_species_list = []
    #对于每一个species的SMILES，如果这个SMILES能转化成rdkit分子，那就将它存进rdkit分子的list
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list

def get_largest_mol(mol_list):
    #给定一个rdkit分子的list，选出含有原子数最多的那个分子，如果一样多就选排在前面的那个
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    #对于每一个mol GetAtoms()数一下原子就好了
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    #得到拥有最多原子的分子的index
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    #返回这个选中的分子
    return mol_list[largest_mol_idx]

def create_all_datasets():
    #### create dataset
    downstream_dir = [
            'bace',
            'bbbp',
            'clintox',
            'esol',
            'freesolv',
            'hiv',
            'lipophilicity',
            'muv',
            'sider',
            'tox21',
            'toxcast'
            ]

    for dataset_name in downstream_dir:
        print(dataset_name)
        root = "dataset/" + dataset_name
        os.makedirs(root + "/processed", exist_ok=True)
        dataset = MoleculeDataset(root, dataset=dataset_name)
        print(dataset)


    dataset = MoleculeDataset(root = "dataset/chembl_filtered", dataset="chembl_filtered")
    print(dataset)
    dataset = MoleculeDataset(root = "dataset/zinc_standard_agent", dataset="zinc_standard_agent")
    print(dataset)


# test MoleculeDataset object
if __name__ == "__main__":

    create_all_datasets()

