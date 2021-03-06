import math
import os
import matplotlib
import pandas as pd

matplotlib.use('SVG')

import mpld3
from mpld3 import plugins
from mpld3.plugins import PointHTMLTooltip

import matplotlib.pyplot as plt


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
        # repeatability (CКО повторяемости)
        self.rep = math.sqrt(self.disp.sum() / self.disp.shape[0])

        self.__filter_grabbs()

        R_l = 2.77 * self.s  # self.s CКО воспроизводимость

        self.__filter_student()
        self.delta_s_l_v = 2 * math.sqrt(
            self.s ** 2 / self.avg.shape[0] + self.delta_0 ** 2 / 3)  # правильность
        self.delta_l_v = 2 * math.sqrt(self.s ** 2 + self.delta_0 ** 2)  # точность

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

            self.s = math.sqrt(
                (1 / (self.avg.shape[0] - 1)) * (self.avg - x_mean).apply(
                    lambda x: x ** 2).sum())
            g_p = (x_p - x_mean) / self.s

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

    def get_data(self):
        return self.data

    def get_avg(self):
        return self.avg

    def get_disp(self):
        return self.disp

    def get_rep(self):
        return self.rep

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
            # print("Контроль повторяемости")
            r_sr = 1.128 * self.sigma_r_l
            r_pr = 2.834 * self.sigma_r_l
            r_d = 3.686 * self.sigma_r_l

            # print("Контроль внутрилаб.прецизионности")
            R_sr = 1.128 * self.sigma_R_l
            R_pr = 2.834 * self.sigma_R_l
            R_d = 3.686 * self.sigma_R_l

        else:  # Методика

            self.sigma_R_l = 0.84 * self.R1 / 2.77
            self.sigma_r_l = 0.84 * self.r / 2.77

            gamma1 = self.sigma_R_l / self.sigma_r_l
            gamma = math.sqrt(gamma1 ** 2 + ((self.n - 1) / self.n))

            # print("Контроль повторяемости")
            r_sr = 1.128 * self.sigma_r_l
            r_pr = 2.834 * self.sigma_r_l
            r_d = 3.686 * self.sigma_r_l

            # print("Контроль внутрилаб.прецизионности")
            R_sr = 1.128 * self.sigma_R_l
            R_pr = 2.834 * self.sigma_R_l
            R_d = 3.686 * self.sigma_R_l

        if m2 == 1:  # Акт
            k = self.delta
        else:  # Расчёт
            delta = 1.96 * self.R2 / 2.77
            k = 0.84 * delta

        k_d = 1.5 * k

        K_k_l = (self.data.mean(axis=1) - self.k)
        r_k_l = (self.data['x1'] - self.data['x2']).abs()
        R_k_l = (
            self.data.mean(axis=1)[1:].reset_index()[0] - self.data.mean(
                axis=1)[
                                                          :-1]).abs()

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
        fig.set_size_inches(18.5, 10.5)
        x = range(1, K_k_l.shape[0] + 1)
        min_x = min(x)
        max_x = max(x)

        ax1.set_title("Контроль повторяемости")
        ax1.grid()
        points = ax1.plot(x, r_k_l, 'ro')
        ax1.plot(x, r_k_l, color='red')
        # ax1.axhline(r_sr, color='green', label="Средняя линия")
        ax1.plot((min_x, max_x), (r_sr, r_sr), color='green',
                 label="Средняя линия")

        # ax1.axhline(r_pr, color='blue', label="Предел предупреждения")
        ax1.plot((min_x, max_x), (r_pr, r_pr), color='blue',
                 label="Предел предупреждения")

        # ax1.axhline(r_d, color='black', label="Предел действия")
        ax1.plot((min_x, max_x), (r_d, r_d), color='black',
                 label="Предел действия")

        print(r_k_l.tolist() + [r_sr, r_pr, r_d])
        # r_sr, r_pr, r_d

        # start, end = ax1.get_ylim()
        # step_count = 10.0
        # step = (end - start) / step_count
        # start = round_down(start, step)
        # end = round_up(end, step)
        ticks = sorted(r_k_l.unique().tolist() + [r_sr, r_pr, r_d])
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(ticks, minor=False)
        ax1.legend()
        #
        # for p in zip(points[0].get_xdata(orig=True),
        #              points[0].get_ydata(orig=True)):
        #     ax1.annotate(p[1], (p[0], p[1]), size=15)

        ax2.set_title("Контроль внутрилабораторной прецизионности")
        ax2.grid()
        points = ax2.plot(x[:-1], R_k_l, 'ro')
        ax2.plot(x[:-1], R_k_l, color='red')
        # ax2.axhline(R_sr, color='green', label="Средняя линия")
        ax2.plot((min_x, max_x), (R_sr, R_sr), color='green',
                 label="Средняя линия")
        # ax2.axhline(R_pr, color='blue', label="Предел предупреждения")
        ax2.plot((min_x, max_x), (R_pr, R_pr), color='blue',
                 label="Предел предупреждения")
        # ax2.axhline(R_d, color='black', label="Предел действия")
        ax2.plot((min_x, max_x), (R_d, R_d), color='black',
                 label="Предел действия")

        ax2.set_xticks(x)
        # start, end = ax2.get_ylim()
        # step_count = 10.0
        # step = (end - start) / step_count
        # start = round_down(start, step)
        # end = round_up(end, step)
        # ax2.set_yticks(np.arange(start, end, step))  # (end - start) / step
        ticks = sorted(R_k_l.unique().tolist() + [R_sr, R_pr, R_d])
        ax2.set_yticks(ticks)
        ax2.set_yticklabels(ticks, minor=False)
        ax2.legend()

        # for p in zip(points[0].get_xdata(orig=True),
        #              points[0].get_ydata(orig=True)):
        #     ax2.annotate(p[1], (p[0], p[1]), size=15)

        ax3.set_title("Контроль точности")
        ax3.grid()
        points = ax3.plot(x, K_k_l, 'ro')
        ax3.plot(x, K_k_l, color='red')
        # ax3.axhline(0, color='green', label="Средняя линия")
        ax3.plot((min_x, max_x), (0, 0), color='green',
                 label="Средняя линия")
        # ax3.axhline(k, color='blue', label="Предел предупреждения")
        ax3.plot((min_x, max_x), (k, k), color='blue',
                 label="Предел предупреждения")
        # ax3.axhline(k_d, color='black', label="Предел действия")
        ax3.plot((min_x, max_x), (k_d, k_d), color='black',
                 label="Предел действия")
        # ax3.axhline(-k, color='blue', label="Предел предупреждения")
        ax3.plot((min_x, max_x), (-k, -k), color='blue',
                 label="Предел предупреждения")
        # ax3.axhline(-k_d, color='black', label="Предел действия")
        ax3.plot((min_x, max_x), (-k_d, -k_d), color='black',
                 label="Предел действия")

        ax3.set_xticks(x)
        # start, end = ax3.get_ylim()
        # step_count = 10.0
        # step = (end - start) / step_count
        # start = round_down(start, step)
        # end = round_up(end, step)
        # ax3.set_yticks(pd.np.arange(start, end, step))
        ticks = sorted(K_k_l.unique().tolist() + [k, -k, k_d, -k_d])
        ax3.set_yticks(ticks)
        ax3.set_yticklabels(ticks, minor=False)
        # ax3.set_yticks([])
        # ax3.tick_params(
        #     axis='y',  # changes apply to the x-axis
        #     which='both',  # both major and minor ticks are affected
        #     left='off',  # ticks along the bottom edge are off
        #     right='off',  # ticks along the bottom edge are off
        #     labelleft='off',  # labels along the bottom edge are off)
        #     labelright='off',  # labels along the bottom edge are off)
        #     gridOn=False,
        #     color='r',
        #     width=0,
        #     tick1On=False,
        #     tick2On=False,
        #     label1On=False,
        #     label2On=False)
        ax3.legend()

        # print(sorted(K_k_l.unique().tolist() + [k, -k, k_d, -k_d]))

        # for p in zip(points[0].get_xdata(orig=True),
        #              points[0].get_ydata(orig=True)):
        #     ax3.annotate(p[1], (p[0], p[1]), size=15)

        # labels = ['<h1>{title}</h1>'.format(title=val) for val in K_k_l.tolist()]
        # # for i in range(10):
        # plugins.connect(fig, PointHTMLTooltip(points[0], labels))
        # print(points[0])
        # plt.show()

        return mpld3.fig_to_html(fig)
