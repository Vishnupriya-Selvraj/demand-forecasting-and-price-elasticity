from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import os
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import kagglehub
from .pipeline import run_forecast_pipeline


def home(request):
    return render(request, 'forecast/home.html')


@csrf_exempt
def run_pipeline(request):
    if request.method == 'POST':
        # optional: allow local CSV upload path via form
        csv_path = request.POST.get('csv_path')
        try:
            horizon_days = int(request.POST.get('horizon_days', '0'))
        except ValueError:
            horizon_days = 0
        if not csv_path:
            dataset_path = kagglehub.dataset_download('beatafaron/fmcg-daily-sales-data-to-2022-2024')
            csv_path = os.path.join(dataset_path, 'FMCG_2022_2024.csv')

        results = run_forecast_pipeline(csv_path, horizon_days=horizon_days)
        request.session['results'] = results
        return redirect('results')
    return redirect('home')


def results(request):
    results = request.session.get('results')
    if not results:
        return redirect('home')
    return render(request, 'forecast/results.html', results)

# Create your views here.
