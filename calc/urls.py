from django.conf.urls import url

from calc import views

urlpatterns = [
    url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'^shewhart$', views.ShewhartView.as_view(), name='shewhart'),
]
