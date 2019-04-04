# der_film

## How to start Rest API Service

**Only** for first start:
```
sudo apt-get install screen
screen -S rest_api_server
```

To start/restart:
```
screen -x rest_api_server
cd ~/rest-api-server
java -jar analyzer-1.0.0.jar --server.port=8000
#press ctrl+a then d
```


## Start analyser script
On Ubuntu for tkinter:
```
sudo apt-get install python3-tk
```
How to start:
```
python film_predictor.py --data-path data --cache-path data\weights.h5 --user-id 150 --recommend 10 --json-path results.json
```
