gamma=0.5
dname="SIGHAN15"  # "SIGHAN15" "HybirdSet", "TtTSet"
dpath="./data/"$dname
bpath="./model/bert/"
cpath="./ckpt/"$dname"_"$gamma"/"

mkdir -p $cpath

python -u evaluation_test.py \
    --bert_path  $bpath/bert.ckpt\
    --bert_vocab $bpath/vocab.txt \
    --train_data $dpath/train.txt \
    --dev_data $dpath/test.txt\
    --test_data $dpath/test.txt\
    --batch_size 10 \
    --lr 1e-5 \
    --dropout 0.1 \
    --number_epoch 10 \
    --gpu_id 0 \
    --print_every 1 \
    --save_every 1 \
    --fine_tune \
    --loss_type FC_FT_CRF\
    --gamma $gamma \
    --model_save_path $cpath \
    --prediction_max_len 128 \
    --dev_eval_path $cpath/dev_pred.txt \
    --final_eval_path $cpath/dev_eval.txt \
    --test_eval_path $cpath/test_%d_pred.txt \
    --l2_lambda 1e-5 \
    --training_max_len 128 \
    --restore_ckpt_path ./ckpt/epoch_483_dev_f1_0.357 \
    --augment_data_file ./tmp.txt
