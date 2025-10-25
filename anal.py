import sys
import struct

import matplotlib.pyplot as plt

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def smooth_percs(arr: np.ndarray, vals: list, window_size: int):
    arr_padded = np.pad(arr, pad_width=(window_size - 1, 0), mode="edge")
    windows = sliding_window_view(arr_padded, window_shape=window_size)
    percs = np.stack([np.mean(windows == val, axis=1) for val in vals], axis=1)
    return percs


def smooth(arr: np.ndarray, window_size: int):
    window = np.ones(window_size) / window_size
    return np.convolve(arr, window, mode="same")


def main():
    if len(sys.argv) < 2:
        print("please specify the .dat file to analyze and graph")
        return

    filename = sys.argv[1]
    with open(filename, "rb") as file:
        data = file.read()

    offset = 0

    def read(fmt, offset, one=False):
        _data = struct.unpack_from(fmt, data, offset)
        if one:
            _data = _data[0]
        return _data, offset + struct.calcsize(fmt)

    version_num, offset = read("<I", offset, one=True)

    games_len, offset = read("<I", offset, one=True)
    games_random_len, offset = read("<I", offset, one=True)
    losses_len, offset = read("<I", offset, one=True)

    # print(version_num, games_len, games_random_len, losses_len)
    print(f"games: {games_len}\trandom games: {games_random_len}\tlosses: {losses_len}")

    games_buf, offset = read(f"<{games_len}f", offset)
    games_random_buf, offset = read(f"<{games_random_len}f", offset)
    losses_buf, offset = read(f"<{losses_len}f", offset)

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
