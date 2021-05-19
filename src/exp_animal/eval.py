import numpy as np
import matplotlib.pyplot as plt
import pickle


def smooth(y, degree):
    y = np.array(y)
    s = []
    for i in range(len(y)):
        if i <=  20:
            s.append(y[i])
        else:
            s.append(np.mean(y[(i-degree):i]))
    return s


plt.figure(figsize=(8,6))

out = pickle.load(open("./out/coco.pkl",'rb'))
train = np.array(out[1])[:,0]
test = np.array(out[2])[:,0]

plt.plot(out[0], smooth(train,10),'-', linewidth=1.5, label='CoCo-training',color='C0')
plt.plot(out[0], smooth(test,10),'--', linewidth=1.5, label='CoCo-testing',color='C0')


out = pickle.load(open("./out/irm.pkl",'rb'))
train = np.array(out[1])[:,0]
test = np.array(out[2])[:,0]

plt.plot(out[0], smooth(train,10),'-', linewidth=1.5, label='IRM-training',color='C1')
plt.plot(out[0], smooth(test,10),'--', linewidth=1.5, label='IRM-testing',color='C1')


out = pickle.load(open("./out/erm.pkl",'rb'))
train = np.array(out[1])[:,0]
test = np.array(out[2])[:,0]

plt.plot(out[0], smooth(train,10),'-', linewidth=1.5, label='ERM-training',color='C2')
plt.plot(out[0], smooth(test,10),'--', linewidth=1.5, label='ERM-testing',color='C2')

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

plt.xlim([91,2000])
plt.ylim([0.4,1])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Iteration',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)

plt.savefig('wild'+'.pdf', bbox_inches='tight')  