python3.7 ood-1.py>2-b5t5.txt
sed -i 's/budget=5/budget=10/' ood-1.py
python3.7 ood-1.py>b10t5.txt
sed -i 's/budget=10/budget=5/' ood-1.py
sed -i "s/split_cfg = {'num_cls_idc':2, 'per_idc_train':5, 'per_idc_val':15, 'per_idc_lake':2500, 'per_ood_train':0, 'per_ood_val':0, 'per_ood_lake':5000}/split_cfg = {'num_cls_idc':2, 'per_idc_train':10, 'per_idc_val':15, 'per_idc_lake':2500, 'per_ood_train':0, 'per_ood_val':0, 'per_ood_lake':5000}/" ood-1.py
python3.7 ood-1.py>b5t10.txt
sed -i 's/budget=5/budget=10/' ood-1.py
python3.7 ood-1.py>b10t10.txt
