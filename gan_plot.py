#!/usr/bin/env python3

import numpy as np

from collections import defaultdict
from matplotlib import pyplot as plt
from os import listdir
from os.path import join, dirname, realpath, exists
from sys import stderr
from typing import Dict, List


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log-path', help='Training log.')
    return parser.parse_args()


def transform_lines(lines: Dict[str, List]):
    plot_points = 100
    value_cnt = len(lines['iter'])

    values_per_point = value_cnt // plot_points
    leftovers = value_cnt % plot_points

    # less plot points than expected -> every point is used
    if values_per_point == 0:
        values_per_point = 1
        leftovers = 0

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
        if leftovers:
            arr = np.array(arr[:-leftovers]).reshape((-1, values_per_point))
        else:
            arr = np.array(arr).reshape((-1, values_per_point))

        if key == 'iter':
            lines[key] = arr.max(axis=1)
        else:
            lines[key] = np.average(arr, axis=1)

    return lines


def parse_log(log):
    lines = defaultdict(list)
    disc_lr_marks, gen_lr_marks = [], []

    with open(log, 'r') as file:
        for line in file:
            if line.startswith('DiscriminatorLearningRateAdapted'):
                disc_lr_marks.append(lines['iter'][-1] if len(lines['iter']) else 0)
                continue
            elif line.startswith('GeneratorLearningRateAdapted'):
                gen_lr_marks.append(lines['iter'][-1] if len(lines['iter']) else 0)
                continue

            fields = line.strip().split(" ")

            lines['iter'].append(int(fields[0].split(":")[-1]))
            lines['gen_train'].append(float(fields[1].split(":")[-1]))
            lines['disc_train'].append(float(fields[2].split(":")[-1]))
            lines['gen_test'].append(float(fields[3].split(":")[-1]))
            lines['disc_test'].append(float(fields[4].split(":")[-1]))
            lines['psnr'].append(float(fields[5].split(":")[-1]))
            lines['diff_psnr'].append(float(fields[6].split(":")[-1]))
            lines['ssim'].append(float(fields[7].split(":")[-1]))
            lines['identity_dist'].append(float(fields[8].split(":")[-1]))

            for i in range(9, len(fields)):
                k, v = fields[i].split(":")
                lines[k].append(float(v))
    return lines, disc_lr_marks, gen_lr_marks


def make_plot(log_path):
    folder = dirname(log_path)
    lines, disc_lr_marks, gen_lr_marks = parse_log(log_path)

    line_width = 1

    if len(lines['iter']) == 0:
        return

    lines = transform_lines(lines)

    chart_fig, (gen_loss_plt, disc_loss_plt) = plt.subplots(2)
    chart_fig2, (psnr_plt, diff_psnr_plt, identity_plt) = plt.subplots(3)

    gen_loss_plt.set_title("Trénovacia a testovacia chyba generátoru")
    # tr_line, = loss_plt.semilogy(lines['iter'], lines['train'], lw=line_width, label='Train loss')
    # ts_line, = loss_plt.semilogy(lines['iter'], lines['test'], lw=line_width, label='Test loss')
    tr_line, = gen_loss_plt.plot(lines['iter'], lines['gen_train'], lw=line_width, label='Train loss')
    ts_line, = gen_loss_plt.plot(lines['iter'], lines['gen_test'], lw=line_width, label='Test loss')

    gen_loss_plt.legend([tr_line, ts_line], ('Train loss', 'Test loss'), loc='upper right')
    for coord in gen_lr_marks:
        gen_loss_plt.axvline(x=coord, color='r')

    disc_loss_plt.set_title("Trénovacia a testovacia chyba diskriminátoru")
    # tr_line, = loss_plt.semilogy(lines['iter'], lines['train'], lw=line_width, label='Train loss')
    # ts_line, = loss_plt.semilogy(lines['iter'], lines['test'], lw=line_width, label='Test loss')
    tr_line, = disc_loss_plt.plot(lines['iter'], lines['disc_train'], lw=line_width, label='Train loss')
    ts_line, = disc_loss_plt.plot(lines['iter'], lines['disc_test'], lw=line_width, label='Test loss')

    disc_loss_plt.legend([tr_line, ts_line], ('Train loss', 'Test loss'), loc='upper right')
    for coord in disc_lr_marks:
        disc_loss_plt.axvline(x=coord, color='r')

    psnr_plt.set_title("PSNR")
    psnr_plt.plot(lines['iter'], lines['psnr'], lw=line_width)

    diff_psnr_plt.set_title("Zlepšenie PSNR voči bil.")
    diff_psnr_plt.plot(lines['iter'], lines['diff_psnr'], lw=line_width)

    identity_plt.set_title("Vzdialenosť identity")
    identity_plt.plot(lines['iter'], lines['identity_dist'], lw=line_width)

    chart_fig.tight_layout()
    chart_fig.savefig(folder + "/plot.png")

    chart_fig2.tight_layout()
    chart_fig2.savefig(folder + "/plot2.png")
    # chart_fig.show()


def main():
    folders = []

    args = get_args()

    if args.log_path is None:
        for folder in listdir("outputs"):
            if not exists(join("outputs", folder, "plot.png")) and "gan" in folder.lower():
                folders.append(join("outputs", folder, "log.txt"))
    else:
        folders = [args.log_path]

    for folder in folders:
        try:
            make_plot(folder)
        except Exception as e:
            print(f"Plotting in folder \"{folder}\" raised error:\n{str(e)}\n", file=stderr)


if __name__ == "__main__":
    main()
