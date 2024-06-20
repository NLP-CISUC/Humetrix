###### HUMOR RECOGNITION ######
mkdir -p data/humor_recognition
python utils/conversion/humor_recognition/clemencio_to_json.py \
    -d ../../Resources/Corpora/Recognizing-Humor-in-Portuguese/Datasets/Balanceados/all.txt

python utils/conversion/humor_recognition/haha_to_json.py \
    -t ../../Resources/Corpora/HAHA@IberLEF2019/haha_2019_train.csv \
    -s ../../Resources/Corpora/HAHA@IberLEF2019/haha_2019_test_gold.csv

python utils/conversion/humor_recognition/haha_to_json.py \
    -t ../../Resources/Corpora/HAHA@IberLEF2021/haha_2021_train.csv \
    -s ../../Resources/Corpora/HAHA@IberLEF2021/haha_2021_test_gold.csv \
    -s ../../Resources/Corpora/HAHA@IberLEF2021/haha_2021_dev_gold.csv

python utils/conversion/humor_recognition/huhu_to_json.py \
    -d ../../Resources/Corpora/HUHU@IberLEF2023/train.csv

python utils/conversion/humor_recognition/humicroedit_to_json.py \
    -t ../../Resources/Corpora/Humicroedit/subtask-1/train.csv \
    -d ../../Resources/Corpora/Humicroedit/subtask-1/dev.csv \
    -s ../../Resources/Corpora/Humicroedit/subtask-1/test.csv

python utils/conversion/humor_recognition/joker_to_json.py \
    -d ../../Resources/Corpora/JOKER-CLEF2023/Task\ 1\ -\ detection/
python utils/conversion/humor_recognition/puntuguese_to_json.py -d ../../Resources/Corpora/BRHuM/data/classification_corpus.json

python utils/conversion/humor_recognition/semeval_to_json.py \
    -td ../../Resources/Corpora/semeval2017_task7/data/test/subtask1-heterographic-test.xml \
    -tl ../../Resources/Corpora/semeval2017_task7/data/test/subtask1-heterographic-test.gold \
    -md ../../Resources/Corpora/semeval2017_task7/data/test/subtask1-homographic-test.xml \
    -ml ../../Resources/Corpora/semeval2017_task7/data/test/subtask1-homographic-test.gold

###### HUMOR INTERPRETATION ######
mkdir -p data/humor_interpretation
python utils/conversion/humor_interpretation/humicroedit_to_json.py \
    -t ../../Resources/Corpora/Humicroedit/subtask-1/train.csv \
    -d ../../Resources/Corpora/Humicroedit/subtask-1/dev.csv \
    -s ../../Resources/Corpora/Humicroedit/subtask-1/test.csv

python utils/conversion/humor_interpretation/puntuguese_to_json.py \
    -c ../../Resources/Corpora/BRHuM/data/puns.json
