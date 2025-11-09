import sys
import struct

import matplotlib.pyplot as plt

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def smooth_percs(arr: np.ndarray, vals: list, window_size: int):
    arr_padded = np.pad(arr, pad_width=(window_size - 1, 0), mode="reflect")
    windows = sliding_window_view(arr_padded, window_shape=window_size)
    percs = np.stack([np.mean(windows == val, axis=1) for val in vals], axis=1)
    return percs


def smooth(arr: np.ndarray, window_size: int):
    # window = np.ones(window_size) / window_size
    # return np.convolve(arr, window, mode="same")
    ret = np.cumsum(arr)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret / window_size


def smooth_rows(arr: np.ndarray, win: int):
    res = np.cumsum(arr, axis=1)
    res[:, win:] = res[:, win:] - res[:, :-win]
    return res / win


def read(data, fmt, offset, one=False):
    _data = struct.unpack_from(fmt, data, offset)
    if one:
        _data = _data[0]
    return _data, offset + struct.calcsize(fmt)


def main():
    if len(sys.argv) < 2:
        print("please specify the .dat file to analyze and graph")
        return

    filename = sys.argv[1]
    with open(filename, "rb") as file:
        data = file.read()

    offset = 0

    version_num, offset = read(data, "<I", offset, one=True)

    if version_num == 2:
        version_2(data)
    elif version_num == 3:
        version_3(data)
    else:
        print("invalid version")


def version_3(data):
    off = 0
    version_num, off = read(data, "<I", off, one=True)
    arrs_len, off = read(data, "<I", off, one=True)
    arr_lens, off = read(data, f"<{arrs_len}I", off)

    print("header:")
    print(f"\tversion_num: {version_num}\tarrs_len: {arrs_len}\tarr_lens: {arr_lens}")

    print("reading hella numbers... ", end="", flush=True)

    # arr_lens = arr_lens[:2]

    arrs = []
    for arr_i, arr_len in enumerate(arr_lens):
        arr, off = read(data, f"<{arr_len}d", off)
        arrs.append(np.array(arr))

    print("done")

    colors_9 = [
        "#AED789",
        "#9FD184",
        "#90CC7F",
        "#82C67A",
        "#ff00ff",  # "#73C175",
        "#64BB70",
        "#56B66B",
        "#47B066",
        "#ffAB62",  # "#39AB62",
    ]
    colors_3 = ["#ee4488", "#444488", "#44ee88"]
    color_1 = "#008888"
    rewards_3 = [-1.0, 0.9, 1.0]

    print("crunching hella numbers... ", end="", flush=True)

    np_games_raw = smooth_percs(arrs[0], rewards_3, 200)
    np_games_avg = smooth_percs(arrs[0], rewards_3, 2000)
    np_games_random_raw = smooth_percs(arrs[1], rewards_3, 50)
    np_games_random_avg = smooth_percs(arrs[1], rewards_3, 500)
    np_losses_raw = smooth(arrs[2], 1)
    np_losses_avg = smooth(arrs[2], 100)
    np_q_values_raw = smooth_rows(arrs[4].reshape([-1, 9]).T, 20)
    np_q_values_avg = smooth_rows(arrs[4].reshape([-1, 9]).T, 1000)

    print("done")

    def plot_smooth_percs(ax, percs_raw, percs_avg, colors):
        n = len(colors)
        xdata = np.arange(percs_raw.shape[0])
        percs_raw_t = percs_raw.T
        percs_avg_t = percs_avg.T
        for i in range(n):
            ax.plot(xdata, percs_raw_t[i], color=colors[i], lw=1, alpha=0.1)
        for i in range(n):
            ax.plot(xdata, percs_avg_t[i], color=colors[i], lw=1)

    def plot_smooth(ax, percs_raw, percs_avg, color):
        xdata = np.arange(percs_raw.shape[0])
        ax.plot(xdata, percs_raw, color=color, lw=1, alpha=0.1)
        ax.plot(xdata, percs_avg, color=color, lw=1)

    def plot_smooth_qs(ax, qs_raw, qs_avg, colors):
        n = len(colors)
        xdata = np.arange(qs_raw.shape[1])
        for i in range(n):
            ax.plot(xdata, qs_raw[i], color=colors[i], lw=1, alpha=0.1)
        for i in range(n):
            ax.plot(xdata, qs_avg[i], color=colors[i], lw=1)

    print("plotting hella numbers... ", end="", flush=True)

    fig, axs = plt.subplots(1, len(arrs) - 1, layout="constrained")
    # fig, axs = plt.subplots(1, 2, layout="constrained")
    axs[0].set_yscale("linear")
    axs[1].set_yscale("linear")
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0, np_games_raw.shape[0])
    axs[1].set_ylim(0, 1)
    axs[1].set_xlim(0, np_games_random_raw.shape[0])
    axs[2].set_ylim(np_losses_raw.min(), np_losses_raw.max())
    axs[2].set_xlim(0, np_losses_raw.shape[0])
    axs[3].set_ylim(np_q_values_raw.min(), np_q_values_raw.max())
    axs[3].set_xlim(0, np_q_values_raw.shape[1])
    # axs[0].set_ylim(0, 1)
    # axs[1].set_ylim(0, 1)

    plot_smooth_percs(axs[0], np_games_raw, np_games_avg, colors_3)
    plot_smooth_percs(axs[1], np_games_random_raw, np_games_random_avg, colors_3)
    plot_smooth(axs[2], np_losses_raw, np_losses_avg, color_1)
    plot_smooth_qs(axs[3], np_q_values_raw, np_q_values_avg, colors_9)

    print("done")

    plt.show()


def version_2(data):
    offset = struct.calcsize("<I")
    games_len, offset = read(data, "<I", offset, one=True)
    games_random_len, offset = read(data, "<I", offset, one=True)
    losses_len, offset = read(data, "<I", offset, one=True)

    # print(version_num, games_len, games_random_len, losses_len)
    print(f"games: {games_len}\trandom games: {games_random_len}\tlosses: {losses_len}")

    games_buf, offset = read(data, f"<{games_len}f", offset)
    games_random_buf, offset = read(data, f"<{games_random_len}f", offset)
    losses_buf, offset = read(data, f"<{losses_len}f", offset)

    # calculate smooth percentages
    games_arr = np.array(games_buf)
    games_random_arr = np.array(games_random_buf)
    losses_arr = np.array(losses_buf)

    results = sorted(set(games_arr))

    print("crunching hella numbers... ", end="", flush=True)

    games_raw = smooth_percs(games_arr, results, 500)
    games_random_raw = smooth_percs(games_random_arr, results, 50)
    losses_raw = smooth(losses_arr, 1)

    games_avg = smooth_percs(games_arr, results, 5000)
    games_random_avg = smooth_percs(games_random_arr, results, 500)
    losses_avg = smooth(losses_arr, 100)

    print("crunched!")

    fig, axs = plt.subplots(1, 3)
    games_ax = axs[0]
    games_random_ax = axs[1]
    losses_ax = axs[2]

    # games_ax.plot(np.arange(games_arr.shape[0]), games_arr)
    # games_random_ax.plot(np.arange(games_random_arr.shape[0]), games_random_arr)
    # losses_ax.plot(np.arange(losses_arr.shape[0]), losses_arr)

    games_xdata = np.arange(games_arr.shape[0])
    games_ax.plot(games_xdata, games_raw.T[0], color="#ee4488", lw=1, alpha=0.1)
    games_ax.plot(games_xdata, games_raw.T[1], color="#444488", lw=1, alpha=0.1)
    games_ax.plot(games_xdata, games_raw.T[2], color="#44ee88", lw=1, alpha=0.1)
    games_ax.plot(games_xdata, games_avg.T[0], color="#ee4488", lw=1)
    games_ax.plot(games_xdata, games_avg.T[1], color="#444488", lw=1)
    games_ax.plot(games_xdata, games_avg.T[2], color="#44ee88", lw=1)
    games_ax.set_xlim(0, games_arr.shape[0])

    games_random_xdata = np.arange(games_random_arr.shape[0])
    games_random_ax.plot(
        games_random_xdata, games_random_raw.T[0], color="#ee4488", lw=1, alpha=0.1
    )
    games_random_ax.plot(
        games_random_xdata, games_random_raw.T[1], color="#444488", lw=1, alpha=0.1
    )
    games_random_ax.plot(
        games_random_xdata, games_random_raw.T[2], color="#44ee88", lw=1, alpha=0.1
    )
    games_random_ax.plot(
        games_random_xdata, games_random_avg.T[0], color="#ee4488", lw=1
    )
    games_random_ax.plot(
        games_random_xdata, games_random_avg.T[1], color="#444488", lw=1
    )
    games_random_ax.plot(
        games_random_xdata, games_random_avg.T[2], color="#44ee88", lw=1
    )
    games_random_ax.set_xlim(0, games_random_arr.shape[0])

    losses_xdata = np.arange(losses_arr.shape[0])
    losses_ax.plot(losses_xdata, losses_raw, color="#008888", lw=1, alpha=0.1)
    losses_ax.plot(losses_xdata, losses_avg, color="#008888", lw=1)

    plt.draw()
    plt.show()


if __name__ == "__main__":
    main()
