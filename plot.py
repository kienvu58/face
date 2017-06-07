import matplotlib.pyplot as plt
import os
import argparse
from utils import *
from config import *


def plot_from_dict(plot_dict, title, xlabel, ylabel, xticks=None, save=False):
    plt.figure()
    for label in plot_dict.keys():
        data = plot_dict[label]
        plt.plot(data[0], data[1], label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if xticks:
        plt.xticks(xticks)
    plt.legend()
    if save:
        filename = "title" + ".png"
        base_path = "figures/"
        if not os.path.isdir(base_path):
            os.makedirs(base_path)
        save_path = os.path.join(base_path, filename)
        plt.savefig(save_path)


def load_result(path_list, folder):
    result = {}
    if not folder:
        for path in path_list:
            basename = os.path.basename(path)
            name, ext = os.path.splitext(basename)
            evaluation = load(path)
            result[name] = evaluation
    else:
        for fn in os.listdir(path_list[0]):
            if not fn.endswith(".dat"):
                continue
            basename = os.path.basename(fn)
            name, ext = os.path.splitext(basename)
            evaluation = load(os.path.join(path_list[0], fn))
            result[name] = evaluation

    return result


def split_result(result):
    """
    Return 2 dicts:
        CMC, ROC: {
            probe: {
                probe_1: {
                    algo_1: ([x], [y]),
                    algo_2: ([x], [y])
                },
                probe_2: {
                    algo_1: ([x], [y]),
                    algo_2: ([x], [y])
                }
            }
            algo: {
                algo_1: {
                    probe_1: ([x], [y]),
                    probe_2: ([x], [y])
                },
                algo_2: {
                    probe_1: ([x], [y]),
                    probe_2: ([x], [y])
                }
            }
        }
    """
    cmc = {
        "probe": {},
        "algo": {}
    }
    roc = {
        "probe": {},
        "algo": {}
    }
    for algo_name in result.keys():
        cmc["algo"][algo_name] = {}
        roc["algo"][algo_name] = {}
    for probe_name in result[algo_name]:
        cmc["probe"][probe_name] = {}
        roc["probe"][probe_name] = {}

    for algo_name in result.keys():
        for probe_name in result[algo_name]:
            cmc_data = result[algo_name][probe_name]["cmc"]
            roc_data = result[algo_name][probe_name]["roc"]
            cmc["probe"][probe_name][algo_name] = cmc_data
            cmc["algo"][algo_name][probe_name] = cmc_data
            roc["probe"][probe_name][algo_name] = roc_data
            roc["algo"][algo_name][probe_name] = roc_data

    return cmc, roc


def plot_cmc(cmc, save=False):
    xlabel = "Rank"
    ylabel = "Cumulative match score"
    for figure_group in cmc.keys():     # 2 groups: probe, algo
        for figure in cmc[figure_group].keys():
            title = "_".join(["cmc", figure_group, figure])
            plot_from_dict(cmc[figure_group][figure],
                           title, xlabel, ylabel, save=save)


def plot_roc(roc, save=False):
    xlabel = "False alarm rate"
    ylabel = "Probability of verification"
    for figure_group in roc.keys():     # 2 groups: probe, algo
        for figure in roc[figure_group].keys():
            title = "_".join(["roc", figure_group, figure])
            plot_from_dict(roc[figure_group][figure],
                           title, xlabel, ylabel, save=save)


def main(args):
    save = args["save"]
    result = load_result(args["list"], folder=args["folder"])
    cmc, roc = split_result(result)
    plot_cmc(cmc, save=save)
    plot_roc(roc, save=save)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot evaluated result.")
    parser.add_argument("list", nargs="+",
                        help="List of evaluations to plot.")
    parser.add_argument("-s", "--save", dest="save", action="store_true",
                        help="Save figures to file.")
    parser.add_argument("-f", "--folder", dest="folder", action="store_true",
                        help="Plot all files in folder")
    main(vars(parser.parse_args()))
    # plot_dict = {
    #     "A": ([0, 1, 2, 3, 4], [0, 1, 4, 9, 16]),
    #     "B": ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    # }
    # plot_from_dict(plot_dict, "test", "X", "Y", True)
    # plt.show()
