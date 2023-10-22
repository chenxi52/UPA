# 对三个变量参数 进行调整，可视化结果
# 直接根据原图数值进行

import matplotlib.pyplot as plt

# 数据点
Y = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
accuracy_visda = [85.7, 87.1, 88.0, 88.5, 88.7, 87.2]
accuracy_office = [89.4, 89.6, 89.9, 89.6, 88.9, 88.6]

# 创建图像
plt.figure(figsize=(8, 6))
fontsize = 20
# 为两个数据集绘制曲线
plt.plot(Y, accuracy_visda, '-o', color='red', label='VisDA-C')
plt.plot(Y, accuracy_office, '-o', color='blue', label='Office')

# 为每个数据点添加标签
for i, txt in enumerate(accuracy_visda):
    plt.annotate(f"{txt}", (Y[i], accuracy_visda[i] - 0.5), ha='center', color='black',fontsize=fontsize)
for i, txt in enumerate(accuracy_office):
    plt.annotate(f"{txt}", (Y[i], accuracy_office[i] + 0.5), ha='center', color='black',fontsize=fontsize)

# 设置图的其他属性
plt.xlabel('γ', fontsize=fontsize )
plt.ylabel('Accuracy (%)',fontsize=fontsize)
plt.ylim(80, 95)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=fontsize, loc='lower right' )
plt.tick_params(labelsize=fontsize)

# 显示图像
plt.tight_layout()
plt.savefig('res/pic/re-sel_r_acc.png')


#####tau
tau = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
accuracy_visda_1 = [88.9, 88.7, 89.0, 88.9, 88.7, 88.5, 88.4]
accuracy_office_1 = [86.7, 89.0, 89.5, 89.8, 89.9, 89.9, 89.6]


# 创建图像
plt.figure(figsize=(8, 6))
# 为两个数据集绘制曲线
plt.plot(tau, accuracy_visda_1, '-o', color='red', label='VisDA-C')
plt.plot(tau, accuracy_office_1, '-o', color='blue', label='Office')

# 为每个数据点添加标签
for i, txt in enumerate(accuracy_visda_1):
    plt.annotate(f"{txt}", (tau[i], accuracy_visda_1[i] - 0.5), ha='center', color='black',fontsize=fontsize)
for i, txt in enumerate(accuracy_office_1):
    plt.annotate(f"{txt}", (tau[i], accuracy_office_1[i] + 0.5), ha='center', color='black',fontsize=fontsize)

# 设置图的其他属性
plt.xlabel('τ', fontsize=fontsize )
plt.ylabel('Accuracy (%)',fontsize=fontsize)
plt.ylim(80, 95)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=fontsize, loc='lower right' )
plt.tick_params(labelsize=fontsize)

# 显示图像
plt.tight_layout()
plt.savefig('res/pic/re-tau_acc.png')




# 图2的数据
k = [2, 4,  6,  8, 10]
accuracy_visda_2 = [88.7, 88.7, 88.9, 88.7,  88.6]
accuracy_office_2 = [89.4, 89.9, 89.1, 89.3, 89.2]

# 创建图像
plt.figure(figsize=(8, 6))
# 为两个数据集绘制曲线
plt.plot(k, accuracy_visda_2, '-o', color='red', label='VisDA-C')
plt.plot(k, accuracy_office_2, '-o', color='blue', label='Office')

# 为每个数据点添加标签
for i, txt in enumerate(accuracy_visda_2):
    plt.annotate(f"{txt}", (k[i], accuracy_visda_2[i] - 0.5), ha='center', color='black',fontsize=fontsize)
for i, txt in enumerate(accuracy_office_2):
    plt.annotate(f"{txt}", (k[i], accuracy_office_2[i] + 0.5), ha='center', color='black',fontsize=fontsize)

# 设置图的其他属性
plt.xlabel('k', fontsize=fontsize )
plt.ylabel('Accuracy (%)',fontsize=fontsize)
plt.ylim(80, 95)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=fontsize, loc='lower right' )
plt.tick_params(labelsize=fontsize)

# 显示图像
plt.tight_layout()
plt.savefig('res/pic/re-k_acc.png')
