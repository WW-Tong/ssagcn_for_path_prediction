import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from metrics import *
from model import SocialSoftAttentionGCN
import copy


def test(KSTEPS=20, scale=0.05):
    global loader_test, model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step = 0
    ade_bigls_batch = []
    for batch in loader_test:
        step += 1
        # Get data
        batch = [torch.Tensor(obs_traj).unsqueeze(0).cuda() for obs_traj in batch]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            non_linear_ped,
            loss_mask,
            V_obs,
            A_obs,
            V_tr,
            vgg_list,
        ) = batch
        obs_traj *= scale
        pred_traj_gt *= scale
        obs_traj_rel *= scale
        pred_traj_gt_rel *= scale
        V_obs *= scale
        V_tr *= scale
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze(), vgg_list)
        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred, V_tr = V_pred[:, :num_of_objs, :], V_tr[:, :num_of_objs, :]

        sx = torch.exp(V_pred[:, :, 2])  # sx
        sy = torch.exp(V_pred[:, :, 3])  # sy
        corr = torch.tanh(V_pred[:, :, 4])  # corr

        cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], 2, 2).cuda()
        cov[:, :, 0, 0] = sx * sx
        cov[:, :, 0, 1] = corr * sx * sy
        cov[:, :, 1, 0] = corr * sx * sy
        cov[:, :, 1, 1] = sy * sy
        mean = V_pred[:, :, 0:2]

        mvnormal = torchdist.MultivariateNormal(mean, cov)


        # Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(
            V_tr.data.cpu().numpy().squeeze().copy(), V_x[-1, :, :].copy()
        )

        raw_data_dict[step] = {}
        raw_data_dict[step]["trgt"] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]["pred"] = []

        for n in range(num_of_objs):
            ade_ls[n] = []
            fde_ls[n] = []

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()

            V_pred_rel_to_abs = V_pred.data.cpu().numpy().squeeze().copy()
            raw_data_dict[step]["pred"].append(copy.deepcopy(V_pred_rel_to_abs))

            for n in range(num_of_objs):
                pred = []
                target = []
                obsrvs = []
                number_of = []
                pred.append(V_pred_rel_to_abs[:, n : n + 1, :])
                target.append(V_y[:, n : n + 1, :])
                obsrvs.append(V_x[:, n : n + 1, :])
                number_of.append(1)
                ade_ls[n].append(ade(pred, target, number_of))
                fde_ls[n].append(fde(pred, target, number_of))
        total = 0
        for n in range(num_of_objs):
            min_index = np.argmin(np.array(ade_ls[n]))
            ade_bigls.append(ade_ls[n][min_index])
            total += min(ade_ls[n])
            fde_bigls.append(fde_ls[n][min_index])
        ade_bigls_batch.append(total / num_of_objs)
    ade_ = sum(ade_bigls) / len(ade_bigls)
    fde_ = sum(fde_bigls) / len(fde_bigls)
    return ade_, fde_, raw_data_dict


if __name__ == "__main__":
    paths = ["./checkpoint/*ssagcn*"]
    KSTEPS = 20

    print("*" * 50)
    print("Number of samples:", KSTEPS)
    print("*" * 50)

    for feta in range(len(paths)):
        ade_ls = []
        fde_ls = []
        path = paths[feta]
        exps = glob.glob(path)
        print("Model being tested are:", exps)

        for exp_path in exps:
            print("*" * 50)
            print("Evaluating model:", exp_path)

            model_path = exp_path + "/val_best.pth"
            args_path = exp_path + "/args.pkl"
            with open(args_path, "rb") as f:
                args = pickle.load(f)
            scale = args.scale
            print("scale=:", scale)
            print("KSTEPS=:", KSTEPS)
            stats = exp_path + "/constant_metrics.pkl"
            with open(stats, "rb") as f:
                cm = pickle.load(f)
            print("Stats:", cm)

            # Data prep
            obs_seq_len = args.obs_seq_len
            pred_seq_len = args.pred_seq_len
            data_set = "./datasets/" + args.dataset + "/"
            loader_test = torch.load("./data/" + args.dataset + "_test.pt")

            # Defining the model
            model = SocialSoftAttentionGCN(
                n_ssagcn=args.n_ssagcn,
                n_txpcnn=args.n_txpcnn,
                output_feat=args.output_size,
                seq_len=args.obs_seq_len,
                kernel_size=args.kernel_size,
                pred_seq_len=args.pred_seq_len,
            ).cuda()
            model.load_state_dict(torch.load(model_path))

            ade_ = 999999
            fde_ = 999999
            print("Testing ....")
            ad, fd, raw_data_dic_ = test(KSTEPS, scale)
            ade_ = min(ade_, ad) / scale
            fde_ = min(fde_, fd) / scale
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            print("ADE:", ade_, " FDE:", fde_)

        print("*" * 50)

        print("Avg ADE:", sum(ade_ls) / len(ade_ls))
        print("Avg FDE:", sum(fde_ls) / len(ade_ls))
