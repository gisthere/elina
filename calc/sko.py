import math
import os
import pandas as pd


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
