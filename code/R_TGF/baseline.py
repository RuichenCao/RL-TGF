
# -*- coding: utf-8 -*-

import numpy as np
import igraph as ig
import time
from collections import defaultdict
import networkx as nx
#from utils import k_triangle ,k_line
import itertools
def k_triangle(g, v_tenuous, k):
    #    assert isinstance(v_tenous, list)
    k_line_all = k_line(g, v_tenuous, k)
    # k_line 组成k_triangle
    k_triangle=0
    tri_num = 0

    for v1, v2 in itertools.combinations(v_tenuous, 2):
        if v1 > v2:
            v1, v2 = v2, v1
        if v2 in k_line_all[v1]:
            k_triangle+=1
        tri_num += len(list(nx.common_neighbors(g, v1, v2)))
    return len(k_line_all), k_triangle,tri_num

def k_line(g, v_tenuous, k):
    '''
    计算稀疏子集 v_tenous 中的 k_line 的数量
    '''
    #assert isinstance(v_tenous, list)
    # 计算所有的k_line
    k_line_all = defaultdict(set)
    for v1, v2 in itertools.combinations(v_tenuous, 2):
        if v1 > v2:
            v1, v2 = v2, v1
        if nx.has_path(g, v1, v2):
            shortest_path_length = nx.shortest_path_length(g, v1, v2)  # 返回边的数量
            if shortest_path_length <= k:
                k_line_all[v1].add(v2)
    #print(k_line_all)
    return k_line_all
def pf(g,v_tenuous,k):
    k_line_all=k_line(g,v_tenuous,k)
    tri_num=0
    for u,d in list(k_line_all.items()):
        for v in d:
            tri_num+=len(list(nx.common_neighbors(g,u,v)))
    return tri_num
def deg(G, v):
    return G.degree(v)/(2*G.ecount()/G.vcount())


def div(G, vertex):
    '''
    ETD算法里中，计算某个节点的feature Ratio
    feature_ratio定义：节点邻居节点的属性并集规模除以子图属性总数
    '''

    LG = set()
    for node in G.neighbors(vertex):
        # if len(G.vs[node]['attr']) == 0:
        #     print(G.vs[node]['id'])
        LG.update(G.vs[node]['attr'])
    if len(LG) == 0:
        return 0
    return len(vertex['attr'])/len(LG)


def ETD(G, p):
    #G.vs['w'] = [deg(G, v)+div(G, v) for v in G.vs]

    G.vs['w'] = [deg(G, node)  for v in G.vs for node in G.neighbors(v)]
    while G.vcount() > 0:
        vlist = [G.vs.select(_degree=min(G.degree()))]
        if len(vlist) > 0:
            wlist = [v['w'] for v in vlist]
            #wlist=vlist
            v = vlist[wlist.index(max(wlist))]
            G.delete_vertices(v)
            if G.ecount() < p:
                break
    return G



def WK(G, H, k, S):
    #G.vs返回图的节点序列
    #G.vs['w']存储节点k跳邻居的数量

    G.vs['w'] = [G.neighborhood_size(v, k) for v in G.vs]
    v = G.vs[0]
    F1 = []
    #start=time.time()
    while G.vcount() > 0:
        F1.append(v['id'])
        if len(F1)>S:
            break

        vlist = [v.index] + G.neighborhood(v, k)  #存储节点v和节点的k跳邻居

         #v的k跳邻居节点，不包括<k跳的节点
        out_node = set(G.neighborhood(v, k))
        # out_node = set(G.neighborhood(v, k)) - \
        #     set(G.neighborhood(v, k-1)) if k > 1 else G.neighbors(v,k)
        updatelist = set([])
        for node in out_node:
            updatelist.update(G.neighborhood(node, k))  #v的k跳邻居节点的邻居节点
        updatelist = updatelist - set(vlist)   #v的k跳邻居节点的邻居节点，不包括k跳邻居节点
        for node in updatelist:
            G.vs[node]['w'] = len(set(G.neighborhood(node, k)) - set(vlist))

        G.delete_vertices(vlist)
        v = np.argsort(np.array(G.vs['w']))
        if G.vcount() > 0:
            G.vs['w'] = [G.neighborhood_size(v_, k) for v_ in G.vs]
            v = G.vs.select(w_eq=min(G.vs['w']))[0]
    
    #print('F1: ', len(F1), F1)

    while len(F1) < S:  # 第二阶段
        candidate_node = {}
        candidate_key = list(set(H.vs['id'])-set(F1))
        for cand in candidate_key: #计算cand的邻居节点和F1中节点重复的个数
            candidate_node[cand] = len(set(H.neighborhood(cand, k)) & set(F1))
        new_cand = sorted(candidate_node.items(),
                          key=lambda item: item[1])[0][0]  
        #将k跳邻居节点和F1中节点重复最少的点加到F1中，保证加入点后使得增加的k-line的数量最少
        F1.append(new_cand)   
        for node in set(H.neighborhood(new_cand, k)) & set(candidate_node.keys()):
            candidate_node[node] += 1
        del candidate_node[new_cand]

    #print('F2: ', len(F1), F1)
    #end = time.time()
    #print('WK run time:', end-start)
    return [H.vs[v_]['node_id'] for v_ in F1]
    #  return F1


#  def triangles(g, k):
#      # g = G.subgraph(nodelist)
#      w = [k_triangle(g, v_, k) for v_ in g.vs]
#      return sum(w)/3

def triangles(G, nodelist):
    s = 0
    for v in nodelist:
        s += G.vs[v]['w']
    return s/3


def TERA(G, O, n, k):
    U = []
    min_triangles = 0
    H = G.vs['id']
    #  other_min = triangles(O.subgraph(G.vs['id']), k)
    other_min = triangles(O, G.vs['id'])
    start = time.time()
    while G.vcount() > n:
        #print(G.vcount(),)
        vlist = G.vs.select(w_eq=min(G.vs['w']))
        #vlist = G.vs.select(w_eq=max(G.vs['w']))
        v = vlist[0]
        G.delete_vertices(v)
        #  triangle =triangles(O.subgraph(G.vs['id']), k)
        triangle = triangles(O, G.vs['id'])
        if G.ecount() == 0:
            if U == []:
                min_triangles = triangle
                U = G.vs['id']
            elif triangle < min_triangles:
                min_triangles = triangle
                U = G.vs['id']
        if triangle < other_min:
            other_min = triangle
            H = G.vs['id']
    end = time.time()
    print('run time:', end-start)

    res = U
    if U == []:
        res = H

    #  return res
    return [O.vs[v_]['node_id'] for v_ in res]


def delta_k(G, u, particle, k):
   
    #计算的是子图的k-density
    

    edges = 0
    node_list = set([u])
    node_list.update(particle)
    for node in node_list:
        edges += len(set(G.neighborhood(node, k)) & node_list) - 1
    return edges/(2*len(particle)**2)


def w(G, particle, Wq):
   
    #计算对一个子图的那个属性相似度
    
    w = 0
    for v in particle:
        wv = len(set(G.vs[v]['attr']) & set(Wq))/len(Wq)
        w += (1 - wv)
    w = w/(len(particle)+1)
    return w


def runETD(G):
    print('\nETD: ')  # utan
    p = 0.1*G.vcount()
    #  U = G.copy()
    start = time.time()
    G = ETD(G, p)
    print('running time: {:.4f}s'.format(time.time()-start))
    v_tenous = G.vs['node_id']
    return v_tenous
    #  u = particle.pop()
    #  Wq = U.vs[u]['attr']
    #  print('k-density: {:.4f}'.format(delta_k(U, U.vs[u], particle, 1)))
    #  print('attr-sim: {:.4f}'.format(w(U, particle, Wq)))

def runWK(G, k, S):
    print('\nWK: ')
    H = G.copy()
    start = time.time()
    v_tenuous = WK(G, H, k, S)
    print('running time: {:.4f}s'.format(time.time()-start))
    #print('v_tenuous',len( v_tenuous))
    return sorted(v_tenuous)

    #  u = H.vs[0]
    #  #  Wq = u['attr']
    #  particle.remove(u['id'])
    #  print('k-density: {:.4f}'.format(delta_k(H,
    #                                           u, particle, k), w(H, particle, Wq)))


def runTERA(G, k, S):
    print('\nTERA: ')  # k-triangle
    start = time.time()
    #G.vs['w'] = [k_triangle(G, v, k) for v in G.vs]
    G.vs['w'] = [G.neighborhood_size(v, k) for v in G.vs]
    OO = G.copy()
    v_tenuous = TERA(G, OO, S, k)
    # s = 0
    # for v in v_tenuous:
    #     s += G.vs[v]['w']
    # print('s'.format((s)))
    #print('running time: {:.4f}s'.format(time.time()-start))
    return sorted(v_tenuous)
    #  u = OO.vs[0]
    #  Wq = u['attr']
    #  print('k-density: {:.4f}'.format(delta_k(OO,
    #                                           u, particle, k), w(OO, particle, Wq)))


def eval(g_networkx, v_tenuous, k):
    n_k_line, n_k_triangle = k_triangle(g_networkx, v_tenuous, k)
    #n_k_density = k_density(g_networkx, v_tenuous, k)

    print('k_line={}'.format(n_k_line),
          'k_triangle={}'.format(n_k_triangle),
          #'k_density={}'.format(n_k_density)
          )
    return n_k_line, n_k_triangle

def attr(features, v_tenuous):
    attr_score=0
    for v1, v2 in itertools.combinations(v_tenuous, 2):
        cnt=0
        for i in range(len(features[v1-1])):
            if features[v1-1][i]==features[v2-1][i]:
                cnt+=1
        #print(cnt,cnt/len(features[v1]))
        attr_score+=cnt/len(features[v1-1])
    return 2*attr_score/(len(v_tenuous)*(len(v_tenuous)-1))
#data_dir = 'data/karate/'
data_dir = '../data/real/'
data_name='citeseer'
G = ig.Graph.Read_Edgelist(data_dir+data_name+'.txt', directed=False)
g = nx.read_edgelist(data_dir+data_name+'.txt', nodetype=int)
#features=np.loadtxt(data_dir+'feature.txt')
#print('num nodes: ', g_networkx.number_of_nodes())

G.vs['node_id'] = range(G.vcount())
G.vs.select(_degree=0).delete() # 删除孤立节点
g1=G.copy()

G.vs['id'] = range(G.vcount())   #G.vs['id']存储节点号

n=g.number_of_nodes()
#s=int(0.05*n)
S=[int(0.03*n),int(0.05*n),int(0.07*n),int(0.1*n),int(0.13*n)]
for s in S:

    r1=runWK(G.copy(),k=1,S=s)
    #print(r1)
    r2=runTERA(G.copy(),k=1,S=s)
    #print(r2)

    #r3=runETD(G.copy())
    #print(r3)
    print(len(r1),k_triangle(g,r1,1))
    print(len(r2),k_triangle(g,r2,2))
#print(len(r3),k_triangle(g,r3,1))
#print(r1)
# filename = '../data/real/'+data_name+'_WK_result.txt'
# with open(filename,'w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
#     for line in r1:
#         f.write(str(line) + '\n')
# filename = '../data/real/'+data_name+'_TERA_result.txt'
# with open(filename,'w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
#     for line in r2:
#         f.write(str(line) + '\n')

# g = nx.read_edgelist(data_dir + 'edgelist.txt', nodetype=int)
#print(triangles(G,r2))
# import matplotlib.pyplot as plt
# g = nx.read_edgelist('./data/lesmis/edgelist.txt', nodetype=int)
# pos = nx.spring_layout(g)
# nx.draw(g,pos,node_color = 'black',node_size = 100)
# plt.savefig('./data/polbooks/polbooks.pdf')
# plt.show()