import requests

resp = requests.post("https://getprediction-tqc5taiqdq-lm.a.run.app", files={'file': open('eight.png', 'rb')})

print(resp.json())
