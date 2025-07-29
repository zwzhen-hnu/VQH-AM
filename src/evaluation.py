import torch
from scipy.stats import pearsonr, spearmanr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluat_metrics(result, label):
    """
    计算 MAE, RMSE, Pearson 相关系数, Spearman 相关系数
    :param result: 预测值 (PyTorch Tensor)
    :param label: 真实值 (PyTorch Tensor)
    :return: 字典 {MAE, RMSE, PearsonR, SpearmanR}
    """
    result = result.cpu().numpy()  # 转为 NumPy 数组
    result = result.squeeze(1)
    label = label.cpu().numpy()

    mae = torch.mean(torch.abs(torch.tensor(result - label))).item()
    rmse = torch.sqrt(torch.mean(torch.tensor((result - label) ** 2))).item()
    pearson_r = pearsonr(result, label)[0]  # 取相关系数
    spearman_r = spearmanr(result, label)[0]

    return {
        "MAE": mae,
        "RMSE": rmse,
        "PearsonR": pearson_r,
        "SpearmanR": spearman_r
    }


def evaluator(model, loader):
    eval_output_list = []
    eval_labels_list = []

    model.eval()

    with torch.no_grad():
        for data_dict in loader:
            data_dict["mutation_graph"] = data_dict["mutation_graph"].to(device)
            data_dict["wild_graph"] = data_dict["wild_graph"].to(device)
            data_dict["ddG"] = data_dict["ddG"].to(device)
            output, loss = model(data_dict["mutation_graph"], data_dict["wild_graph"], data_dict["ddG"])

            eval_output_list.append(output.detach().cpu())
            eval_labels_list.append(data_dict["ddG"].detach().cpu())

    return evaluat_metrics(torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0))

