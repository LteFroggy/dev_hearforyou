import requests

address = "http://127.0.0.1:8000/"
target = ""

path = address + target

result = requests.get(address)

print(result.json())