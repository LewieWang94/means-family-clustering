from sklearn import datasets


def generator(features, centers):
    output = open('cluster_dim_' + str(features) + '_' + str(centers) + '.txt', 'w')
    iris = datasets.make_blobs(n_samples=10000,
                               n_features=features,
                               centers=centers,
                               cluster_std=1.0)[0]
    for i in iris:
        for j in i:
            output.write(str(j))
            output.write(' ')
        output.write('\n')


f_sweep = [2, 16, 32, 64]
c_sweep = [2, 4, 6, 8, 10]

for f in f_sweep:
    for c in c_sweep:
        generator(f, c)
