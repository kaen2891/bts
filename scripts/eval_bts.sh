
MODEL="laion/clap-htsat-unfused"
SEED="1"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="BTS_eval"
        CUDA_VISIBLE_DEVICES=0 python main.py --tag $TAG \
                                        --dataset icbhi \
                                        --seed $s \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 50 \
                                        --batch_size 8 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --sample_rate 48000 \
                                        --model $m \
                                        --model_type ClapModel \
                                        --meta_mode all \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --method ce \
                                        --print_freq 100 \
                                        --eval \
                                        --pretrained \
                                        --pretrained_ckpt /home2/jw/workspace/mcl/BTS/save/icbhi_laion/clap-htsat-unfused_ce_all_BTS_bs8_lr5e-5_ep50_seed1_meta_all/best.pth

    done
done
