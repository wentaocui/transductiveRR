from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('tkagg')

def visualize(train_data_before, test_data_before, train_data_after, test_data_after, args):
    z = np.concatenate((train_data_before, train_data_after, test_data_before, test_data_after), axis=0)
    z_2d = TSNE(n_components=2).fit_transform(z)

    n_support = args.meta_val_way * args.eval_shot
    n_query = args.meta_val_way * args.meta_val_query

    label_before = ['C1', 'C2', 'C3', 'C4', 'C5']
    y_before = [str for str in label_before for iter in range(int(args.eval_shot))] + [str for str in label_before for iter in range(int(n_query / args.meta_val_way))]
    label_after = ['C1 t', 'C2 t', 'C3 t', 'C4 t', 'C5 t']
    y_after = [str for str in label_after for iter in range(int(args.eval_shot))] + [str for str in label_after for iter in range(int(n_query / args.meta_val_way))]

    y_all = y_before + y_after
    plt.close('all')
    sns.scatterplot(x=z_2d[:, 0], y=z_2d[:, 1], hue=y_all, legend='full')
    plt.show()


    # s_2d = z_2d[0:n_support]
    # sw_2d = z_2d[n_support:n_support * 2]
    # q_2d = z_2d[n_support * 2:n_support * 2 + n_query]
    # qw_2d = z_2d[n_support * 2 + n_query:]

    # label_spt = ['C1 s', 'C2 s', 'C3 s', 'C4 s', 'C5 s']
    # y_support = [str for str in label_spt for iter in range(int(args.eval_shot))]
    # label_qry = ['C1 q', 'C2 q', 'C3 q', 'C4 q', 'C5 q']
    # y_query = [str for str in label_qry for iter in range(int(n_query / args.meta_val_way))]
    #
    # label_spt_t = ['C1 st', 'C2 st', 'C3 st', 'C4 st', 'C5 st']
    # y_support_t = [str for str in label_spt_t for iter in range(int(args.eval_shot))]
    # label_qry_t = ['C1 qt', 'C2 qt', 'C3 qt', 'C4 qt', 'C5 qt']
    # y_query_t = [str for str in label_qry_t for iter in range(int(n_query / args.meta_val_way))]
    #
    # plt.close('all')
    # sns.scatterplot(x=s_2d[:, 0], y=s_2d[:, 1], hue=y_support, legend='full')
    # plt.show()
    # sns.scatterplot(x=q_2d[:, 0], y=q_2d[:, 1], hue=y_query, legend='full')
    # plt.show()
    # sns.scatterplot(x=sw_2d[:, 0], y=sw_2d[:, 1], hue=y_support_t, legend='full')
    # plt.show()
    # sns.scatterplot(x=qw_2d[:, 0], y=qw_2d[:, 1], hue=y_query_t, legend='full')
    # plt.show()

