import pickle as pkl
import os
import os.path as op
import sys
import matplotlib.pyplot as plt
import numpy as np


def loss_line_graph(loss_store_folder, mode="depart", save_fig_name="tmp.jpg"):
    """loss折线图"""
    loss_G = []
    loss_G_identity_A = []
    loss_G_identity_B = []
    loss_G_GAN_A2B = []
    loss_G_GAN_B2A = []
    loss_G_cycle_ABA = []
    loss_G_cycle_BAB = []
    loss_D_A = []
    loss_D_B = []

    for i in range(200):
        file_path = op.join(loss_store_folder, "losses_record_{}.pkl".format(i))
        losses_epoch = pkl.load(open(file_path, "rb"))
        loss_G_sum = 0
        loss_G_identity_A_sum = 0
        loss_G_identity_B_sum = 0
        loss_G_GAN_A2B_sum = 0
        loss_G_GAN_B2A_sum = 0
        loss_G_cycle_ABA_sum = 0
        loss_G_cycle_BAB_sum = 0
        loss_D_A_sum = 0
        loss_D_B_sum = 0
        for losses_batch in losses_epoch:
            loss_G_sum += losses_batch["loss_G"].cpu().detach()
            loss_G_identity_A_sum += losses_batch["loss_G_identity_A"].cpu().detach()
            loss_G_identity_B_sum += losses_batch["loss_G_identity_B"].cpu().detach()
            loss_G_GAN_A2B_sum += losses_batch["loss_G_GAN_A2B"].cpu().detach()
            loss_G_GAN_B2A_sum += losses_batch["loss_G_GAN_B2A"].cpu().detach()
            loss_G_cycle_ABA_sum += losses_batch["loss_G_cycle_ABA"].cpu().detach()
            loss_G_cycle_BAB_sum += losses_batch["loss_G_cycle_BAB"].cpu().detach()
            loss_D_A_sum += losses_batch["loss_D_A"].cpu().detach()
            loss_D_B_sum += losses_batch["loss_D"].cpu().detach()  # NOTES loss_D_B就是记录时候的loss_D，记录的时候打错了
        loss_G.append(loss_G_sum / 200)
        loss_G_identity_A.append(loss_G_identity_A_sum / 200)
        loss_G_identity_B.append(loss_G_identity_B_sum / 200)
        loss_G_GAN_A2B.append(loss_G_GAN_A2B_sum / 200)
        loss_G_GAN_B2A.append(loss_G_GAN_B2A_sum / 200)
        loss_G_cycle_ABA.append(loss_G_cycle_ABA_sum / 200)
        loss_G_cycle_BAB.append(loss_G_cycle_BAB_sum / 200)
        loss_D_A.append(loss_D_A_sum / 200)
        loss_D_B.append(loss_D_B_sum / 200)

    x_axis = np.array(range(1, 201))
    if mode == "depart":
        plt.plot(x_axis, loss_G, label="loss_G")
        plt.show()
        plt.plot(x_axis, loss_G_identity_A, label="loss_G_identity_A")
        plt.show()
        plt.plot(x_axis, loss_G_identity_B, label="loss_G_identity_B")
        plt.show()
        plt.plot(x_axis, loss_G_GAN_A2B, label="loss_G_GAN_A2B")  # 这个loss一直上升
        plt.show()
        plt.plot(x_axis, loss_G_GAN_B2A, label="loss_G_GAN_B2A")  # 这个loss一直上升
        plt.show()
        plt.plot(x_axis, loss_G_cycle_ABA, label="loss_G_cycle_ABA")
        plt.show()
        plt.plot(x_axis, loss_G_cycle_BAB, label="loss_G_cycle_BAB")
        plt.show()
        plt.plot(x_axis, loss_D_A, label="loss_D_A")
        plt.show()
        plt.plot(x_axis, loss_D_B, label="loss_D_B")
        plt.show()
    elif mode == "combine":
        plt.plot(x_axis, loss_G, label="loss_G")
        plt.plot(x_axis, loss_G_identity_A, label="loss_G_identity_A")
        plt.plot(x_axis, loss_G_identity_B, label="loss_G_identity_B")
        # plt.plot(x_axis, loss_G_GAN_A2B, label="loss_G_GAN_A2B")
        # plt.plot(x_axis, loss_G_GAN_B2A, label="loss_G_GAN_B2A")
        plt.plot(x_axis, loss_G_cycle_ABA, label="loss_G_cycle_ABA")
        plt.plot(x_axis, loss_G_cycle_BAB, label="loss_G_cycle_BAB")
        plt.plot(x_axis, loss_D_A, label="loss_D_A")
        plt.plot(x_axis, loss_D_B, label="loss_D_B")
        plt.show()
        plt.savefig("plt_jpgs/{}".format(save_fig_name))
    elif mode == "classify":
        loss_G_indentity = [loss_G_identity_A[ind] + loss_G_identity_B[ind] for ind in range(len(loss_G_identity_B))]
        loss_G_gan = [loss_G_GAN_A2B[ind] + loss_G_GAN_B2A[ind] for ind in range(len(loss_G_GAN_B2A))]
        loss_G_cycle = [loss_G_cycle_ABA[ind] + loss_G_cycle_BAB[ind] for ind in range(len(loss_G_cycle_BAB))]
        loss_D = [loss_D_A[ind] + loss_D_B[ind] for ind in range(len(loss_D_B))]

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # plt.plot(x_axis, loss_G, label="loss_G")
        # plt.plot(x_axis, loss_G_indentity, label="loss_G_indentity")
        # plt.plot(x_axis, loss_G_gan, label="loss_G_gan")
        # plt.plot(x_axis, loss_G_cycle, label="loss_G_cycle")
        plt.plot(x_axis, loss_D, label="loss_G")
        plt.legend()  # 展示小窗
        plt.savefig("../plt_jpgs/{}".format(save_fig_name))
        plt.show()


if __name__ == '__main__':
    # data_folder = "output_41_49_gen"
    # data_folder = "output_41_49_unet"
    # data_folder = "output_51_98_gen"
    data_folder = "output_51_98_unet"
    # desc = "gen"
    desc = "disc"
    save_fig_name = "loss_{}_".format(desc) + data_folder
    # 训练的数据
    loss_path = r"../../tmp_store/{}".format(data_folder)

    loss_line_graph(loss_path, mode="classify", save_fig_name=save_fig_name)
