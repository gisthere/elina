import math
import os

import matplotlib.pyplot as plt
import mpld3
import pandas as pd
import numpy as np


def round_up(x, a):
    return math.ceil(x / a) * a


def round_down(x, a):
    return math.floor(x / a) * a


class Sko:
    table_dir = os.path.dirname(__file__) + '/misc'

    def __init__(self, data, n, k, delta_0):
        self.data = data
        self.n = n
        self.k = k
        self.delta_0 = delta_0
        self.avg = self.data.mean(axis=1)
        self.disp = self.data.var(axis=1)

        self.__filter_kohern()
        # repeatability
        self.rep = math.sqrt(self.disp.sum() / self.disp.shape[0])

        self.__filter_grabbs()
        R_l = 2.77 * self.s

        self.__filter_student()
        self.delta_s_l_v = 2 * math.sqrt(
            self.s ** 2 / self.avg.shape[0] + self.delta_0 ** 2 / 3)
        self.delta_l_v = 2 * math.sqrt(self.s ** 2 + self.delta_0 ** 2)

    def __filter_kohern(self):
        k_table = pd.read_excel(Sko.table_dir + "/Kohern.xlsx", index_col=0)
        g = k_table.loc[self.disp.shape[0], self.n]
        kohern = self.disp.max() / self.disp.sum()

        while kohern > g:
            self.avg.drop(self.disp.idxmax(), inplace=True)
            self.disp.drop(self.disp.idxmax(), inplace=True)

            g = k_table.loc[self.disp.shape[0], self.n]
            kohern = self.disp.max() / self.disp.sum()

    def __filter_grabbs(self):
        g_table = pd.read_excel(Sko.table_dir + "/Grabbs.xlsx", index_col=0,
                                header=None)
        const = g_table.loc[self.disp.shape[0], 1]

        x_mean = self.avg.mean()
        x_p = self.avg.max()
        self.s = math.sqrt(
            (1 / (self.avg.shape[0] - 1)) * (self.avg - x_mean).apply(
                lambda x: x ** 2).sum())
        g_p = (x_p - x_mean) / self.s

        while g_p > const:
            self.avg.drop(self.disp.idxmax(), inplace=True)
            self.disp.drop(self.disp.idxmax(), inplace=True)
            x_mean = self.avg.mean()
            x_p = self.avg.max()
            s = math.sqrt(
                (1 / (self.avg.shape[0] - 1)) * (self.avg - x_mean).apply(
                    lambda x: x ** 2).sum())
            g_p = (x_p - x_mean) / s

    def __filter_student(self):
        s_table = pd.read_excel(Sko.table_dir + "/Student.xlsx", index_col=0,
                                header=None)
        x_mean = self.avg.mean()
        theta_l = x_mean - self.k
        f = s_table.loc[self.avg.shape[0] - 1, 1]
        t = math.fabs(theta_l) / math.sqrt(
            self.s ** 2 / self.avg.shape[0] + self.delta_0 ** 2 / 3)

        while t > f:
            self.avg.drop(self.disp.idxmax(), inplace=True)
            self.disp.drop(self.disp.idxmax(), inplace=True)
            f = s_table.loc[self.avg.shape[0] - 1, 1]
            t = math.fabs(theta_l) / math.sqrt(
                self.s ** 2 / self.avg.shape[0] + self.delta_0 ** 2 / 3)

    def get_delta_s_l_v(self):
        return self.delta_s_l_v

    def get_delta_l_v(self):
        return self.delta_l_v

    def get_s(self):
        return self.s


class ShewhartMap:
    def __init__(self, data, n, k,
                 sigma_R_l=None, sigma_r_l=None,
                 R1=None, R2=None, r=None, delta=None):
        self.data = data

        self.n = n
        self.k = k
        self.sigma_R_l = sigma_R_l
        self.sigma_r_l = sigma_r_l
        self.R1 = R1
        self.R2 = R2
        self.r = r
        self.delta = delta

    def get_map(self, m1, m2):
        if m1 == 1:  # Акт

            gamma1 = self.sigma_R_l / self.sigma_r_l
            gamma = math.sqrt(gamma1 ** 2 + ((self.n - 1) / self.n))
            print("Гамма =", gamma, sep=" ")
            # print("Контроль повторяемости")
            r_sr = 1.128 * 0.01 * self.sigma_r_l
            r_pr = 2.834 * 0.01 * self.sigma_r_l
            r_d = 3.686 * 0.01 * self.sigma_r_l

            # print("Контроль внутрилаб.прецизионности")
            R_sr = 1.128 * 0.01 * self.sigma_R_l
            R_pr = 2.834 * 0.01 * self.sigma_R_l
            R_d = 3.686 * 0.01 * self.sigma_R_l

        else:  # Методика

            self.sigma_R_l = 0.84 * self.R1 / 2.77
            self.sigma_r_l = 0.84 * self.r / 2.77

            gamma1 = self.sigma_R_l / self.sigma_r_l
            gamma = math.sqrt(gamma1 ** 2 + ((self.n - 1) / self.n))

            # print("Контроль повторяемости")
            r_sr = 1.128 * 0.01 * self.sigma_r_l
            r_pr = 2.834 * 0.01 * self.sigma_r_l
            r_d = 3.686 * 0.01 * self.sigma_r_l

            # print("Контроль внутрилаб.прецизионности")
            R_sr = 1.128 * 0.01 * self.sigma_R_l
            R_pr = 2.834 * 0.01 * self.sigma_R_l
            R_d = 3.686 * 0.01 * self.sigma_R_l

        if m2 == 1:  # Расчёт
            delta = 1.96 * self.R2 * 2.77
            k = 0.01 * 0.84 * delta
        else:
            k = 0.01 * self.delta  # Акт

        k_d = 1.5 * k

        x1 = self.data['x1'].tolist()
        x2 = self.data['x2'].tolist()

        x1 = [
            0.015,
            0.0158,
            0.0162,
            0.0159,
            0.0179,
            0.015,
            0.0175,
            0.0122,
            0.0138,
            0.0076,
        ]

        x2 = [
            0.017,
            0.0136,
            0.0166,
            0.0179,
            0.0169,
            0.0144,
            0.0185,
            0.0148,
            0.0120,
            0.0125,
        ]

        average_X = []
        for i in range(len(x1)):
            n3 = (x1[i] + x2[i]) / 2
            average_X.append(n3)
        # результат контроля точности

        K_k_l = []
        for i in range(len(x1)):
            o = (average_X[i] - self.k) / self.k
            K_k_l.append(o)

        # результат контроля повторяемости

        r_k_l = []
        for i in range(len(x1)):
            q = round(abs(x1[i] - x2[i]) / average_X[i], 4)
            r_k_l.append(q)
        print(r_k_l)

        ###########################################
        # контроль прецизионности

        R_k_l = []
        x_sr = sum(average_X) / len(average_X)
        for i in range(len(average_X) - 1):
            w = abs(average_X[i] - average_X[i + 1]) / x_sr
            R_k_l.append(w)

        # контроль повторяемости

        r1_d = []
        r_pr1 = []
        r_sr1 = []
        for i in range(1, len(x1) + 1):
            r1_d.append(r_d)
            r_pr1.append(r_pr)
            r_sr1.append(r_sr)

        x = np.arange(len(x1))
        y = []
        c = []
        c1 = []
        c12 = []
        for i in range(len(r1_d)):
            x[i] += 1
            h = r1_d[i]
            h1 = r_pr1[i]
            h12 = r_sr1[i]
            z = r_k_l[i]
            c.append(h)
            c1.append(h1)
            c12.append(h12)
            y.append(z)

        ####################################

        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
        f.set_size_inches(18.5, 10.5)

        # ax1.set_xlabel("Номер контрольной процедуры")
        # ax1.set_ylabel("Результат контрольной процедуры")
        ax1.set_title("Контроль точности с ОК")
        ax1.grid()
        ax1.plot(x, y, 'ro')
        ax1.plot(x, y, color='red')
        ax1.plot(x, c, color='green', label="Предел действия")
        ax1.plot(x, c1, color='grey', label="Предел предупреждения")
        ax1.plot(x, c12, color='black', label="Средняя линия")
        ax1.set_xticks(x)
        start, end = ax1.get_ylim()
        step = 0.05
        start = round_down(start, step)
        end = round_up(end, step)
        ax1.set_yticks(np.arange(start, end, step))
        ax1.legend(loc=2)

        #####################################

        # контроль воспроизводимости

        R1_d = []
        R_pr1 = []
        R_sr1 = []
        for i in range(1, len(x1) + 1):
            R1_d.append(R_d)
            R_pr1.append(R_pr)
            R_sr1.append(R_sr)
            # i = i + 1

        x = np.arange(len(x1) - 1)
        y = []
        c = []
        c1 = []
        c12 = []
        for i in range(len(R1_d) - 1):
            x[i] = x[i] + 1
            h = R1_d[i]
            h1 = R_pr1[i]
            h12 = R_sr1[i]
            z = R_k_l[i]
            c.append(h)
            c1.append(h1)
            c12.append(h12)
            y.append(z)

        ax2.set_title("Контроль внутрилабораторной прецизионности")
        ax2.grid()
        ax2.plot(x, y, 'ro')
        ax2.plot(x, y, color='red')
        ax2.plot(x, c, color='green', label="Предел действия")
        ax2.plot(x, c1, color='grey', label="Предел предупреждения")
        ax2.plot(x, c12, color='black', label="Средняя линия")
        ax2.set_xticks(x)
        start, end = ax2.get_ylim()
        step = 0.05
        start = round_down(start, step)
        end = round_up(end, step)
        ax2.set_yticks(np.arange(start, end, step))
        ax2.legend(loc=2)

        #####################################

        # Точность

        k1 = []
        k12 = []
        kd = []
        k_pr = []
        k_pr1 = []
        for i in range(1, len(x1) + 1):
            k1.append(k_d)
            k12.append(-1 * k_d)
            kd.append(0)
            k_pr.append(k)
            k_pr1.append(-1 * k)
            # i = i + 1

        x = np.arange(len(x1))
        y = []
        c = []
        c1 = []
        c12 = []
        c123 = []
        c1234 = []
        for i in range(len(K_k_l)):
            x[i] = x[i] + 1
            h = k1[i]
            h1 = k12[i]
            h12 = kd[i]
            h123 = k_pr[i]
            h1234 = k_pr1[i]
            z = K_k_l[i]
            # i = i + 1
            c.append(h)
            c1.append(h1)
            c12.append(h12)
            c123.append(h123)
            c1234.append(h1234)
            y.append(z)

        ax3.set_title("Контроль повторяемости")
        ax3.grid()
        ax3.plot(x, y, 'ro')
        ax3.plot(x, y, color='red')
        ax3.plot(x, c, color='green')
        ax3.plot(x, c, color='green', label="Предел действия")
        ax3.plot(x, c1, color='grey', label="Предел предупреждения")
        ax3.plot(x, c12, color='black', label="Средняя линия")
        ax3.plot(x, c1234, color='grey')

        ax3.set_xticks(x)
        start, end = ax3.get_ylim()
        step = 0.05
        start = round_down(start, step)
        end = round_up(end, step)
        ax3.set_yticks(np.arange(start, end, step))
        ax3.legend(loc=2)

        # ax = f.add_subplot(111, frameon=False)
        # # hide tick and tick label of the big axes
        # ax.tick_params(labelcolor='none', top='off', bottom='off', left='off',
        #                 right='off')
        # ax.grid(False)
        # ax.set_xlabel("common X")
        # ax.set_ylabel("common Y")

        #
        # plt.title("Контроль внутрилабораторной прецизионности")
        # plt.grid()
        # ax = plt.axes()
        # ax.set_yticks(np.arange(0, 1, 0.05))
        # ax.set_xticks(np.arange(0, len(x1) + 1, 1))
        # plt.axis([0, len(x1), -1 * min(R_k_l),
        #           max(max(c) + 0.1, max(R_k_l) + 0.5 * max(R_k_l))])
        # plt.plot(x, y, 'ro')
        # plt.plot(x, y, color='red')
        # plt.plot(x, c, color='green')
        # plt.plot(x, c1, color='grey')
        # plt.plot(x, c12, color='black')
        # plt.annotate('Предел действия', xy=(5, (1.04 * R_d)),
        #              xytext=(5, (1.04 * R_d)))
        # plt.annotate('Предел предупреждения', xy=(5, (1.04 * R_pr)),
        #              xytext=(5, (1.04 * R_pr)))
        # plt.annotate('Средняя линия', xy=(5, (1.07 * R_sr)),
        #              xytext=(5, (1.07 * R_sr)))
        # plt.ylabel("Результат контрольной процедуры")
        # plt.xlabel("Номер контрольной процедуры")
        # plt.savefig("plotsample1.png")

        ####################################

        print(x1)
        print(x2)

        return mpld3.fig_to_html(f)
