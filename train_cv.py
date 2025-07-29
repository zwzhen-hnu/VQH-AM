param = {
    "dataset": "SHS27k",
    "split_mode": "random",
    "input_dim": 44,
    "output_dim": 7,
    "prot_hidden_dim": 256,
    "prot_num_layers": 4,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "max_epoch": 10000,
    "batch_size": 8,
    "dropout_ratio": 0.0,
    "commitment_cost": 0.25,
    "num_embeddings": 1024,
    "mask_ratio": 0.15,
    "sce_scale": 1.5,
    "mask_loss": 1,
    "seed": 114514,
    "num_folds": 5,
}

import torch
import copy
from src.evaluation import evaluat_metrics, evaluator
from src.datasets import skempi2_dataset, collate
import random
import numpy as np
from sklearn.model_selection import KFold


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(param["seed"])

processed_dir = "/root/autodl-tmp/processed_data/"
# dataset
data = skempi2_dataset(processed_dir)
dataset_size = len(data)

# 交叉验证
kf = KFold(n_splits=param["num_folds"], shuffle=True, random_state=param["seed"])

from torch.utils.data import DataLoader, Subset
from src.my_log import getLogger
from src.models import ddG

logger = getLogger("ddG")

# 记录所有折的测试结果
final_results = {
    "MAE": [],
    "RMSE": [],
    "PearsonR": [],
    "SpearmanR": []
}

for fold, (train_idx, test_idx) in enumerate(kf.split(range(dataset_size))):
    logger.info(f"Fold {fold + 1}/{param['num_folds']}")

    # 划分训练集和测试集
    train_dataset = Subset(data, train_idx)
    test_dataset = Subset(data, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=param["batch_size"], shuffle=True, pin_memory=True,
                              collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=param["batch_size"], shuffle=False, pin_memory=True,
                             collate_fn=collate)

    # 重新初始化模型
    model = ddG(param)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param["learning_rate"],
        weight_decay=param["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    early_stop = 0
    best_mae, best_rmse, best_pearson_r, best_spearman_r = 100, 100, 100, 100

    # 训练循环
    for epoch in range(param["max_epoch"]):
        model.train()
        loss_sum, mae, rmse, pearson_r, spearman_r = 0.0, 0.0, 0.0, 0.0, 0.0

        for data_dict in train_loader:
            data_dict["mutation_graph"] = data_dict["mutation_graph"].to(device)
            data_dict["wild_graph"] = data_dict["wild_graph"].to(device)
            data_dict["ddG"] = data_dict["ddG"].to(device)
            output, loss = model(data_dict["mutation_graph"], data_dict["wild_graph"], data_dict["ddG"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            score = evaluat_metrics(output.detach().cpu(), data_dict["ddG"].detach().cpu())
            mae += score["MAE"]
            rmse += score["RMSE"]
            pearson_r += score["PearsonR"]
            spearman_r += score["SpearmanR"]

        scheduler.step(loss_sum / len(train_loader))

        # 计算测试集表现
        test_score = evaluator(model, test_loader)
        if epoch > 9 and test_score["MAE"] < best_mae:
            best_mae = test_score["MAE"]
            best_rmse = test_score["RMSE"]
            best_pearson_r = test_score["PearsonR"]
            best_spearman_r = test_score["SpearmanR"]
            early_stop = 0
        else:
            early_stop = early_stop + 1

        if early_stop == 15:
            break

        logger.info(
            "Fold: {}, Epoch: {}, Train Loss: {:.5f} | Train: {:.4f},{:.4f},{:.4f},{:.4f}, Test: {:.4f},{:.4f},{:.4f},{:.4f}".format(
                fold + 1, epoch, loss_sum / len(train_loader),
                mae / len(train_loader), rmse / len(train_loader),
                pearson_r / len(train_loader), spearman_r / len(train_loader),
                test_score["MAE"], test_score["RMSE"], test_score["PearsonR"], test_score["SpearmanR"]
            )
        )

    # 记录本折测试结果
    logger.info(f"Fold {fold + 1}/{param['num_folds']} : {best_mae}, {best_rmse}, {best_pearson_r}, {best_spearman_r}")
    final_results["MAE"].append(best_mae)
    final_results["RMSE"].append(best_rmse)
    final_results["PearsonR"].append(best_pearson_r)
    final_results["SpearmanR"].append(best_spearman_r)

# 计算 5 折交叉验证的平均测试结果
avg_results = {metric: np.mean(values) for metric, values in final_results.items()}
std_results = {metric: np.std(values) for metric, values in final_results.items()}

logger.info("===== Final Cross-Validation Results =====")
logger.info("MAE: {:.4f} ± {:.4f}".format(avg_results["MAE"], std_results["MAE"]))
logger.info("RMSE: {:.4f} ± {:.4f}".format(avg_results["RMSE"], std_results["RMSE"]))
logger.info("PearsonR: {:.4f} ± {:.4f}".format(avg_results["PearsonR"], std_results["PearsonR"]))
logger.info("SpearmanR: {:.4f} ± {:.4f}".format(avg_results["SpearmanR"], std_results["SpearmanR"]))