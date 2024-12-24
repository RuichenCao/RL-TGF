#在cora数据集上子图规模和PF指标之间的关系
import numpy as np
import matplotlib.pyplot as plt
name_list=['3%','5%','7%','10%','13%']
y1=[322,1376,1668,2484,3247]
y2=[852,1464,2108,3109,4194]
y3=[114,183,291,474,677]
y4=[10,10,20,93,114]
#plt.plot(x,y,'r-',marker='o',linewidth=2)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

x = np.arange(len(name_list))
total_width,n=0.8,4
width=0.2
#plt.bar(x,y1,width,color='#000000',label='WK')
plt.bar(x - 1.5*width, y1, width=width,color='#F08080', label='WK')
plt.bar(x - 0.5*width,y2,width=width,color='#FFDAB9',label='TERA')
plt.bar(x+0.5*width,y3,width=width,color='#FF8C00',label='GNNM')
plt.bar(x+1.5*width,y4,width=width,color='#ADD8E6',label='R_TGF')
plt.xlabel('graph size',font1)
plt.xticks(x, labels=name_list,fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel('|PF|',font1)
#plt.grid(which='minor')  #不显示网格线
#ax.set_xticklabels(name_list)
plt.legend(prop = {'family': 'Times New Roman', 'weight': 'normal','size':10.5})
plt.savefig('cora_PF.pdf',dpi=2000)
plt.show()




#在citeseer数据集上子图规模和PF指标之间的关系
import numpy as np
name_list=['3%','5%','7%','10%','13%']
y1=[265,461,696,1368,2387]
y2=[2332,4063,5643,7344,8539]
y3=[225,299,582,853,951]
y4=[8,29,48,154,262]
#plt.plot(x,y,'r-',marker='o',linewidth=2)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

x = np.arange(len(name_list))
total_width,n=0.8,4
width=0.2
#plt.bar(x,y1,width,color='#000000',label='WK')
plt.bar(x - 1.5*width, y1, width=width,color='#F08080', label='WK')
plt.bar(x - 0.5*width,y2,width=width,color='#FFDAB9',label='TERA')
plt.bar(x+0.5*width,y3,width=width,color='#FF8C00',label='GNNM')
plt.bar(x+1.5*width,y4,width=width,color='#ADD8E6',label='R_TGF')
plt.xlabel('graph size',font1)
plt.xticks(x, labels=name_list,fontsize=12)
plt.yticks(fontsize=12)
plt.xticks(x, labels=name_list)
plt.ylabel('|PF|',font1)
#plt.grid(which='minor')  #不显示网格线
#ax.set_xticklabels(name_list)
plt.legend(prop = {'family': 'Times New Roman', 'weight': 'normal','size':10.5})
plt.savefig('citeseer_PF.pdf',dpi=2000)
plt.show()


#运行时间
x=['Synthetic_1000','Synthetic_5000','Synthetic_10000','Synthetic_50000','Synthetic_100000']
y1=[2.432,9.064,28.665,1117.785,4290.1317]
y2=[0.623,24.413,108.7,2193.043,12906.681]
y3=[0.049,0.494,4.042,19.310,263.992]
y4=[0.0079,0.707,1.28,15.203,38.76]
#plt.plot(x,y,'r-',marker='o',linewidth=2)
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}

ax=plt.gca()
plt.tick_params(labelsize=12)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.plot(x,y1,c='#A0522D',marker='.',label='WK',linewidth=2.5,markersize=10)
plt.plot(x,y2,c='#1E90FF',marker='v',label='TERA',linewidth=2.5,markersize=7)
plt.plot(x,y3,c='#4682B4',marker='*',label='GNNM',linewidth=2.5,markersize=10)
plt.plot(x,y4,c='#FF4500',marker='^',label='R_TGF',linewidth=2.5,markersize=7)
plt.xticks(rotation=10)
#plt.xlabel('datasets',font1)
plt.ylabel('running time/s',font1)
plt.grid(which='minor')  #不显示网格线
plt.legend(prop = {'family': 'Times New Roman', 'weight': 'normal','size':10.5})
plt.savefig('rf_running_time_2.pdf',dpi=2000)

plt.show()
