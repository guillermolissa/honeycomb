!/bin/sh

echo "*******************   PROCESSING DATA  *******************"
echo
#python src/data/make_dataset.py --input_file data/credit_card_clients.xls --output_file data/credit_card_clients.csv
echo
echo "*****************  SPLIT TRAIN/TEST ****************"
echo
#python src/fold/hold_out.py --input_file data/credit_card_clients.csv --test_size 0.3
echo
echo "*****************  MAKE K FOLDS  ****************"
echo
#python src/fold/kfolds.py --input_file data/train.csv --kfold 5
python src/fold/kfolds.py --input_file data/credit_card_clients.csv --kfold 5
# echo
# echo "*****************  MAKE FEATURES ENGINEER ****************"
# echo
python features/build_features.py --input_file credit_card_clients.csv --output_file credit_card_clients_fe.csv
# echo
echo "******************* TRAINING CV MODELS *******************"
echo
echo
python src/models/train.py --file_name data/credit_card_clients_fe.csv --model id3_gini --metric roc_auc 
echo
echo
# #python src/models/train.py --file_name data/credit_card_clients_fe.csv --model id3_entropy --metric roc_auc 
# #echo
# #echo
# python src/models/train.py --file_name data/credit_card_clients_fe.csv --model rf --metric roc_auc 
# echo
# echo
# python src/models/train.py --file_name data/credit_card_clients_fe.csv --model lgb --metric roc_auc 
# echo
# echo
# python src/models/train.py --file_name data/credit_card_clients_fe.csv --model xgb --metric roc_auc 
# echo
# echo
# echo "************************  END   ************************"