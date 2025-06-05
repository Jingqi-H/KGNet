from utils.visulize import plot_lr
import matplotlib.pyplot as plt
import os


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(i_iter, base_lr, total_niters, lr_power):
    lr = lr_poly(base_lr, i_iter, total_niters, lr_power)
    # optimizer.param_groups[0]['lr'] = lr
    # if len(optimizer.param_groups) > 1:
    #     optimizer.param_groups[1]['lr'] = lr * 10
    return lr


if __name__ == '__main__':
    """
    查看经过300ep的lr decay之后的学习率是多大的
    得到的结果是4.83307811961276e-06
    因此我可以用这个学习率在ep300之后接着训练
    """

    max_epoch = 200
    bs = 4
    # len(data_loader) = 802
    total_niters = max_epoch * 802
    learning_rate = 1e-3
    lr_epoch = []
    num_step = 160  # 0-159
    for per_epoch in range(1, max_epoch+1):
        lr_list = []
        for step in range(num_step):
            current_idx = (per_epoch - 1) * 802 + step
            adjust_lr = adjust_learning_rate(current_idx, base_lr=learning_rate,
                                             total_niters=total_niters,
                                             lr_power=0.9)
            lr_list.append(adjust_lr)

        lr_epoch += lr_list

    print(lr_epoch)
    # plot_lr(lr_epoch)
    # plt.savefig('demo_lr.png')
    # plt.show()


#  ep800 1.999180264547569e-06
#  ep500 3.051827205882701e-06
#  ep300 4.83307811961276e-06
#  ep200 6.961550021729523e-06

