# try:
    # now it can reach class A of file_a.py in folder a 
    # by relative import
# except (ModuleNotFoundError, ImportError) as e:
# 	print("{} fileure".format(type(e)))
# else:
# 	print("Import succeeded")
from django.shortcuts import render
from django.http import Http404
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.core import serializers
from django.conf import settings
import json 
import _pickle as pickle
#from .predictRF import predictIris
# from ..iris.predictRF import predictIris
import os
import sys
lib_path = os.path.abspath(os.path.join('../iris'))
print(lib_path)
sys.path.append(lib_path)
print(sys.path)
# import
from predictRF import predictIris


@api_view(["POST"])
def testThang(request):
	try: 
		data = json.loads(request.body)
		#weight = str(height*10)
		# with open('iris_rfc.pkl', 'rb') as f:
		# 	my_random_forest = pickle.load(f)

		#my_random_forest = pickle.load(open("iris_rfc.pkl","rb"))
		#result = my_random_forest.predict([[5.84,3.0,3.75,1.1]])
		#result = my_random_forest.predict([[request['sl'],request['sw'],request['pl'],request['pw']]])[0]
		ir = predictIris(data['sl'],data['sw'],data['pl'],data['pw'])


		#return JsonResponse("Hahaaah response"+weight+"kg",safe=False)
		return Response({'result':ir.get_predict()})
	except ValueError as e:
		# return JsonResponse(e.args[0],status.HTTP_400_BAD_REQUEST)
		return Response(status=status.HTTP_400_BAD_REQUEST)