from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('run/', views.run_pipeline, name='run_pipeline'),
    path('results/', views.results, name='results'),
]


