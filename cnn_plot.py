#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
from os import listdir
from typing import Dict, List


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_path', help='Training log.')
    return parser.parse_args()


def transform_lines(lines: Dict[str, List]):
    plot_points = 100
    value_cnt = len(lines['iter'])

    values_per_point = value_cnt // plot_points
    leftovers = value_cnt % plot_points

    # average values into points
    for key, arr in lines.items():
        if type(arr[0]) == int:
            arr_t = np.int
        elif type(arr[0]) == float:
            arr_t = np.float
        else:
            raise Exception("Unexpected value type!")

        # leave out few last values, so that each point in plot
        # averages the same number of values
        arr = np.array(arr[:-leftovers], dtype=arr_t).reshape((-1, values_per_point))

        if key == 'iter':
            lines[key] = arr.max(axis=1)
        else:
            lines[key] = np.average(arr, axis=1)

    return lines


def make_plot(folder):
    log = folder + "/log.txt"
    #    if args.log_path is None else args.log_path
    lines = {'iter': [], 'train': [], 'test': [], 'psnr': [], 'diff_psnr': []}
    lr_marks = []

    with open(log, 'r') as file:
        for line in file:
            if line.startswith('LearningRateAdapted'):
                lr_marks.append(lines['iter'][-1])
                continue

            fields = line.split(" ")

            lines['iter'].append(int(fields[0].split(":")[-1]))
            lines['train'].append(float(fields[1].split(":")[-1]))
            lines['test'].append(float(fields[2].split(":")[-1]))
            lines['psnr'].append(float(fields[3].split(":")[-1]))
            lines['diff_psnr'].append(float(fields[4].split(":")[-1]))

    line_width = 1

    if len(lines['iter']) == 0:
        return

    lines = transform_lines(lines)

    chart_fig, (loss_plt, psnr_plt, diff_psnr_plt) = plt.subplots(3)
    loss_plt.set_title("Trénovacia a testovacia chyba")
    #tr_line, = loss_plt.semilogy(lines['iter'], lines['train'], lw=line_width, label='Train loss')
    #ts_line, = loss_plt.semilogy(lines['iter'], lines['test'], lw=line_width, label='Test loss')
    tr_line, = loss_plt.plot(lines['iter'], lines['train'], lw=line_width, label='Train loss')
    ts_line, = loss_plt.plot(lines['iter'], lines['test'], lw=line_width, label='Test loss')

    for coord in lr_marks:
        loss_plt.axvline(x=coord, color='r')

    loss_plt.legend([tr_line, ts_line], ('Train loss', 'Test loss'), loc='upper right')

    psnr_plt.set_title("PSNR")
    psnr_plt.plot(lines['iter'], lines['psnr'], lw=line_width)

    diff_psnr_plt.set_title("Zlepšenie PSNR voči bil.")
    diff_psnr_plt.plot(lines['iter'], lines['diff_psnr'], lw=line_width)

    chart_fig.tight_layout()
    chart_fig.savefig(folder + "/plot.png")
    #chart_fig.show()


def main():
    folders = []
    for folder in listdir("outputs"):
        if "plot.png" not in listdir("outputs/" + folder):
            folders.append("./outputs/" + folder)

    for folder in folders:
        make_plot(folder)


    #plt.show()

if __name__ == "__main__":
    main()
