# AirWrite

## 1) Install
pip install -r requirements.txt

## 2) Record data
Record character samples into:
data/samples/<LABEL>/*.npy

Example:
python src/record_data.py --label A --count 80
python src/record_data.py --label B --count 80

## 3) Train
python src/train.py

## 4) Run
python src/run_app.py

Controls:
ESC quit | c clear | s add space | b backspacepython src/record_data.py --label A --count 80# test--zone
