from django.template.response import TemplateResponse
from django.views.generic import TemplateView
import pandas as pd

from calc.sko import Sko


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

        response = TemplateResponse(request, "result.html",
                                    dict(mr=mr, s=s,
                                         delta_s_l_v=delta_s_l_v,
                                         delta_l_v=delta_l_v))

        return response
