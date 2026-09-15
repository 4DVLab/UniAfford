export CUDA_VISIBLE_DEVICES=0,1,2,3


python validate.py \
--checkpoint_path ../ckpt/UniAfford-5B/best_fsdp.pth \
--dataset_dir ../datasets/ReasonAff_converted_test \
--batch_size 1