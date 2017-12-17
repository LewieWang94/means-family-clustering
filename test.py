# directory = 'data/'
# path = 'semeion.data'
#
# data = open(directory+path, 'r')
# output = open('data/cluster_semeion.txt', 'w')
#
# for i in data.readlines():
#     arr = i.split()
#     arr[len(arr)-1] = arr[len(arr)-1][:-1]
#     for j in arr:
#         output.write(j)
#         output.write(' ')
#     output.write('\n')

# from gmeans import GMeans
# import pandas as pd
# from matplotlib import pyplot as plt
# import seaborn as sbn
# from sklearn.cluster import MiniBatchKMeans
# from sklearn import datasets
#
# iris = datasets.load_iris().data
# gmeans = GMeans(random_state=1010, strictness=4)
# gmeans.fit(iris)
# plot_data = pd.DataFrame(iris[:, 0:2])
# plot_data.columns = ['x', 'y']
# plot_data['labels_gmeans'] = gmeans.labels_
#
# km = MiniBatchKMeans()
# km.fit(iris)
# plot_data['labels_km'] = km.labels_
# sbn.lmplot(x='x', y='y', data=plot_data, hue='labels_gmeans', fit_reg=False)
# sbn.lmplot(x='x', y='y', data=plot_data, hue='labels_km', fit_reg=False)
# plt.show()


# from sklearn import datasets
# output = open('data/cluster_dim_32_10.txt', 'w')
# iris = datasets.make_blobs(n_samples=10000,
#                            n_features=32,
#                            centers=10,
#                            cluster_std=1.0)[0]
# for i in iris:
#     for j in i:
#         output.write(str(j))
#         output.write(' ')
#     output.write('\n')


import numpy as np

a = [[1, 2, 3], [4, 5, 6]]
print(type(np.asarray(a)))
print(np.asarray(a))
