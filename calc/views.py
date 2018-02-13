from django.http import HttpResponse
from django.template.response import TemplateResponse
from django.views.generic import TemplateView
import pandas as pd

from calc.sko import Sko, ShewhartMap


class IndexView(TemplateView):
    def get(self, request, *args, **kwargs):
        response = TemplateResponse(request, "index.html")
        return response

    def post(self, request, *args, **kwargs):

        x1 = [value for key, value in sorted(request.POST.items()) if
              key.startswith("x1_")]
        x1 = pd.Series(x1, name='x1', dtype='float32')
        x2 = [value for key, value in sorted(request.POST.items()) if
              key.startswith("x2_")]
        x2 = pd.Series(x2, name='x2', dtype='float32')
        data = pd.concat([x1, x2], axis=1)

        r = request.POST['r']
        R = request.POST['R']
        mr = request.POST['mr']
        n = int(request.POST['n'])
        k = float(request.POST['k'])
        delta_0 = float(request.POST['delta_0'])

        sko = Sko(data, n, k, delta_0)
        s = sko.get_s()
        delta_s_l_v = sko.get_delta_s_l_v()
        delta_l_v = sko.get_delta_l_v()
        # mr,s,delta_s_l_v,delta_l_v

        if request.POST['action'] == 'calc':
            response = TemplateResponse(request, "result.html",
                                        dict(mr=mr, s=s,
                                             delta_s_l_v=delta_s_l_v,
                                             delta_l_v=delta_l_v))
        else:

            d = pd.concat([sko.get_data(),
                           sko.get_avg(),
                           sko.get_disp()], axis=1)
            response = TemplateResponse(request, "act.html",
                                        dict(data=d,
                                             s=sko.get_s(),
                                             rep=sko.get_rep(),
                                             delta_s_l_v=delta_s_l_v,
                                             delta_l_v=delta_l_v))

        return response


class ShewhartView(TemplateView):
    def get(self, request, *args, **kwargs):
        response = TemplateResponse(request, "shewhart.html")
        return response

    def post(self, request, *args, **kwargs):
        # filter
        x1 = [(key, value) for key, value in request.POST.items() if
              key.startswith("x1_")]
        # sort
        x1 = sorted(x1, key=lambda x: int(x[0][3:]))
        # get only values
        x1 = map(lambda x: x[1], x1)

        x1 = pd.Series(x1, name='x1', dtype='float32')

        x2 = [(key, value) for key, value in request.POST.items() if
              key.startswith("x2_")]

        x2 = sorted(x2, key=lambda x: int(x[0][3:]))
        x2 = map(lambda x: x[1], x2)
        x2 = pd.Series(x2, name='x2', dtype='float32')
        data = pd.concat([x1, x2], axis=1)

        # print(sorted(request.POST.items()))

        n = int(request.POST['n'])
        k = float(request.POST['k'])

        sigma_R_l = request.POST.get('sigma_R_l', None)
        sigma_R_l = float(sigma_R_l) if sigma_R_l else None

        sigma_r_l = request.POST.get('sigma_r_l', None)
        sigma_r_l = float(sigma_r_l) if sigma_R_l else None

        R1 = request.POST.get('R1', None)
        R1 = float(R1) if R1 else None

        r = request.POST.get('r', None)
        r = float(r) if r else None

        delta = request.POST.get('delta', None)
        delta = float(delta) if delta else None

        R2 = request.POST.get('R2', None)
        R2 = float(R2) if R2 else None

        if request.POST.get('rb1', None) == 'radio1':
            m1 = 1
        else:
            m1 = 2

        if request.POST.get('rb2', None) == 'radio3':
            m2 = 1
        else:
            m2 = 2

        sm = ShewhartMap(data, n, k, sigma_R_l=sigma_R_l,
                         sigma_r_l=sigma_r_l,
                         R1=R1, R2=R2, r=r, delta=delta)
        response = HttpResponse(
            "<center>" + sm.get_map(m1, m2) + "</center>")

        return response
