import numpy as np

from read_spectrum import DEFAULT_SPECTRUM, read_spectrum


def main():
    spectrum = read_spectrum(DEFAULT_SPECTRUM)
    freq = spectrum["freq"]
    s_xy = spectrum["S_xy"]

    selected = (freq >= 3000) & (freq <= 5000)
    std = np.std(s_xy[selected])
    mean=np.mean(s_xy[selected])
    print(mean)
    print(std)

    # 原来这里的平均值才是标准差吗？？？这个dat文件有点迷惑。



if __name__ == "__main__":
    main()
