from django.template.response import TemplateResponse
from django.views.generic import TemplateView


class IndexView(TemplateView):
    template_name = "index.html"

    def get(self, request, *args, **kwargs):
        response = TemplateResponse(request, self.template_name)
        return response

    def post(self, request, *args, **kwargs):
        response = TemplateResponse(request, self.template_name)
        return response
