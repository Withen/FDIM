import time
#import numba as nb
from pandas import *
from collections import defaultdict
#import numpy as NP
import sys

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() 
                            if len(locs)>0)

def compute_no_trivial_closure(x,listofcol):#list list
	global dictClosure
	x_str=','.join(sorted(x))
	for item in x:
		listofcol.remove(item)
	for y in listofcol:
		if y not in dictClosure[x_str] :
			if validfd(x,y):#x-list y-元素
				dictClosure[x_str].append(y)

def obtaint_FD_and_Key(x):#list
	global finallistofFDs
	#global finallistofEQs
	global finallistofKEYs
	global dictClosure
	#global listofcolumns
	x_str=','.join(sorted(x))
	for rh in dictClosure[x_str]:#元素 in list
		finallistofFDs.append([x,rh])# list->元素
	if check_superkey(x):#list
		finallistofKEYs.append(x)

def obtain_EQSet(level, listofcols):#list[list] list
	global dictCplus
	global finallistofFDs
	global listofcolumns
	li = [t for t in dictClosure.keys() if dictClosure[t]]
	for x in level: #list in list[list]
		#x_str=','.join(x)
		x_set=set(x)
		#for t in dictClosure.keys():#t是元素
			#if dictClosure[t]:
				#li.append(t)
		for lh in li:#元素 in list
			# lh_set = set(lh.split(","))
			if set(lh.split(",")) != x_set:
				z = x_set.intersection(set(lh.split(",")))  # 交集set
				#x_z = list(set(x) - z)  # list
				x_z=x_set-z
				# filter(None,x_z)#去掉空元素
				x_z = ','.join(x_z)
				#lh_z = list(set(lh.split(",")) - z)  # list
				lh_z = set(lh.split(","))- z
				# filter(None, lh_z)#去掉空元素
				lh_z = ','.join(lh_z)
				if x_z in dictClosure[lh] and lh_z in dictClosure[','.join(x)]:
					finallistofEQs.append([x, list(lh.split(","))])  # list,list

def validfd(x,y):#x是左手边list，y是右手边元素
	if x=='' or y=='': return False
	ey = computeE(x)#x是左手边list
	temp=[item for item in x ]
	#for item in x:
	#	temp.append(item)
	temp.append(y)
	sorted(temp)
	yy=[]
	yy.append(y)
	generate_closure(temp,x,yy)#list list list
	stripped_product(temp,x,yy)#list list list
	eyz = computeE(temp)
	if ey == eyz:
		return True
	else:
		return False

def computeE(x):#x-list
	global totaltuples
	global dictpartitions
	doublenorm = 0
	x_str = ','.join(sorted(x))#list转换成string
	for i in dictpartitions[x_str]:
		doublenorm = doublenorm + len(i)
	e = (doublenorm-len(dictpartitions[ ','.join(sorted(x))]))/float(totaltuples)
	return e

def check_superkey(x):#list
    global dictpartitions
    if ((dictpartitions[','.join(sorted(x))] == [[]]) or (dictpartitions[','.join(sorted(x))] == [])):
        return True
    else:
        return False

def prune_candidates(level):#list[list]
	global finallistofEQs
	global finallistofKEYs
	le=level[:]
	for x in le:  # list in list[list]
		if 	x in finallistofKEYs:#list
			level.remove(x)
		else:
			for y in level:
				if x != y and (([x, y] in finallistofEQs) or ([y, x] in finallistofEQs)):
					level.remove(x)
					break

def generate_closure(x, y, z):#list list list
	global dictClosure
	global dictpartitions
	temp=[]
	y_str=','.join(sorted(y))#string类型的y
	if dictClosure[y_str]:
		#for item in dictClosure[y_str]:
		#	temp.append(item)
		temp+=[item for item in dictClosure[y_str]]
	z_str = ','.join(sorted(z))  # string类型的y
	if dictClosure[z_str]:
		#for item in dictClosure[z_str]:
			#temp.append(item)
		temp += [item for item in dictClosure[z_str]]
	#temp=list(set(temp))
	#x_str = ','.join(sorted(x))
	dictClosure[','.join(sorted(x))]=list(set(temp))

def generate_candidates(level):#list[list]
	global finallistofFDs
	global finallistofKEYs
	nextlevel = []
	for i in range(0, len(level)):  # pick an element
		for j in range(i + 1, len(level)):  # compare it to every element that comes after it.
			if ((not level[i] == level[j]) and level[i][0:-1] == level[j][0:-1]):  # i.e. line 2 and 3
				x = list(level[i]) #list
				x.append(level[j][-1] ) #合成的candidate
				x.sort()
				flag = True
				for a in x:  # x-list
					t=x[:]
					t.remove(a)
					if not ( t in level):
						flag = False
				for k in x:# x-list
					re=x[:]
					re.remove(k)
					re = ''.join(re)
					if k in dictClosure[re]:
						flag = False
				if flag == True:
					#if ','.join(x) not in dictpartitions.keys():
					#	stripped_product(x, level[i], level[j])
					#if ','.join(x) not in dictClosure.keys():
					#	generate_closure(x, level[i], level[j])
					if check_superkey(x):
						finallistofKEYs.append(x)
					else:
						nextlevel.append(x)
	return nextlevel

def stripped_product(x,y,z):#list list list
	global dictpartitions
	global tableT
	tableS = ['']*len(tableT)
	#y_str = ','.join(sorted(y))
	#x_str = ','.join(sorted(x))
	#z_str=','.join(sorted(z))
	partitionY = dictpartitions[','.join(sorted(y))] # partitionY is a list of lists, each list is an equivalence class
	partitionZ = dictpartitions[','.join(sorted(z))]
	partitionofx = [] # line 1
	for i in range(len(partitionY)):
		for t in partitionY[i]:
			tableT[t] = i
		tableS[i]=''
	for i in range(len(partitionZ)):
		for t in partitionZ[i]:
			if ( not (tableT[t] == 'NULL')):
				tableS[tableT[t]] = sorted(list(set(tableS[tableT[t]]) | set([t]))) 
		for t in partitionZ[i]:
			if (not (tableT[t] == 'NULL')) and len(tableS[tableT[t]])>= 2 :
				partitionofx.append(tableS[tableT[t]]) 
			if not (tableT[t] == 'NULL'): tableS[tableT[t]]=''
	for i in range(len(partitionY)): # line 11
		for t in partitionY[i]: # line 12
			tableT[t]='NULL'
	dictpartitions[','.join(sorted(x))] = partitionofx

def computeSingletonPartitions(listofcols):#list[list]
	global data2D
	global dictpartitions	
	for a in listofcols:#list in list[list]
		#a_str=','.join(a)
		dictpartitions[','.join(a)]=[]
		for element in list_duplicates(data2D[','.join(a)].tolist()): # list_duplicates returns 2-tuples, where 1st is a value, and 2nd is a list of indices where that value occurs
			if len(element[1])>1: # ignore singleton equivalence classes
				dictpartitions[','.join(a)].append(element[1])
    
#------------------------------------------------------- START ---------------------------------------------------
t0=time.time()
if len(sys.argv) > 1:
    infile=str(sys.argv[1])# this would be e.g. "testdata.csv"
data2D = read_csv(infile)
totaltuples = len(data2D.index)
listofcolumns = list(data2D.columns.values) # returns ['A', 'B', 'C', 'D', .....]
listofcolumns_list=[] # returns [['A'], ['B'], ['C'], ['D'], .....]
listofcolumns_list=[[item] for item in listofcolumns ]
#for item in listofcolumns:#元素 in list
#	temp_list=[]
#	temp_list.append(item)
#	listofcolumns_list.append(temp_list)
tableT = ['NULL']*totaltuples # this is for the table T used in the function stripped_product
L0 = []
dictClosure = defaultdict(list)
for x in listofcolumns:
	dictClosure[x]=[]
dictpartitions = {} # maps 'stringslikethis' to a list of lists, each of which contains indices
computeSingletonPartitions(listofcolumns_list)#list[list]
finallistofFDs=[]# list->元素
finallistofEQs=[]# list<->list
finallistofKEYs=[]#list
L1=listofcolumns_list[:]  # list[list]  -每个candidate都是list类型
l=1
L = [L0,L1]#level0和level1
while (not (L[l] == [])):
	for x in L[l]:#list in list[list]
		compute_no_trivial_closure(x, listofcolumns[:])#list list
		obtaint_FD_and_Key(x)#list
	obtain_EQSet(L[l], listofcolumns[:])#list[list] list
	prune_candidates(L[l])#list[list]
	temp = generate_candidates(L[l])#list[list]
	L.append(temp)
	l = l + 1
print("List of all FDs: ", finallistofFDs)
print("Total number of FDs found: ", len(finallistofFDs))
print("List of all eqs: ", finallistofEQs)
print(format(time.time()-t0))