# rm -r /mnt/data2/akshit/Organ/OOD_3/weights/*
# CUDA_VISIBLE_DEVICES=3 python3.7 ood-3.py>ab30t10.txt
# sed -i 's/from trust.utils.organmnist3 import load_dataset_custom_aid as load_dataset_custom/from trust.utils.organmnist3 import load_dataset_custom_cid as load_dataset_custom/' ood-3.py
# sed -i 's/idclass = "a"/idclass = "c"/' ood-3.py
rm -r /mnt/data2/akshit/Organ/OOD_3/weights/*
CUDA_VISIBLE_DEVICES=3 python3.7 ood-3.py>cb30t15.txt
sed -i 's/from trust.utils.organmnist3 import load_dataset_custom_cid as load_dataset_custom/from trust.utils.organmnist3 import load_dataset_custom_sid as load_dataset_custom/' ood-3.py
sed -i 's/idclass = "c"/idclass = "s"/' ood-3.py
rm -r /mnt/data2/akshit/Organ/OOD_3/weights/*
CUDA_VISIBLE_DEVICES=3 python3.7 ood-3.py>sb30t15.txt

# sed -i 's/budget=75/budget=100/' ood-1.py
# CUDA_VISIBLE_DEVICES=1,2 python3.7 ood-1.py>b100t50.txt
# sed -i 's/budget=100/budget=50/' ood-1.py
# sed -i "s/split_cfg = {'num_cls_idc':2, 'per_idc_train':50, 'per_idc_val':15, 'per_idc_lake':2500, 'per_ood_train':0, 'per_ood_val':0, 'per_ood_lake':5000}/split_cfg = {'num_cls_idc':2, 'per_idc_train':75, 'per_idc_val':15, 'per_idc_lake':2500, 'per_ood_train':0, 'per_ood_val':0, 'per_ood_lake':5000}/" ood-1.py
# CUDA_VISIBLE_DEVICES=1,2 python3.7 ood-1.py>b50t75.txt
# sed -i 's/budget=50/budget=75/' ood-1.py
# CUDA_VISIBLE_DEVICES=1,2 python3.7 ood-1.py>b75t75.txt
# sed -i 's/budget=75/budget=100/' ood-1.py
# CUDA_VISIBLE_DEVICES=1,2 python3.7 ood-1.py>b100t75.txt
# sed -i 's/budget=100/budget=50/' ood-1.py
# sed -i "s/split_cfg = {'num_cls_idc':2, 'per_idc_train':75, 'per_idc_val':15, 'per_idc_lake':2500, 'per_ood_train':0, 'per_ood_val':0, 'per_ood_lake':5000}/split_cfg = {'num_cls_idc':2, 'per_idc_train':100, 'per_idc_val':15, 'per_idc_lake':2500, 'per_ood_train':0, 'per_ood_val':0, 'per_ood_lake':5000}/" ood-1.py
# CUDA_VISIBLE_DEVICES=1,2 python3.7 ood-1.py>b50t100.txt
# sed -i 's/budget=50/budget=75/' ood-1.py
# CUDA_VISIBLE_DEVICES=1,2 python3.7 ood-1.py>b75t100.txt
# sed -i 's/budget=75/budget=100/' ood-1.py
# CUDA_VISIBLE_DEVICES=1,2 python3.7 ood-1.py>b100t100.txt
