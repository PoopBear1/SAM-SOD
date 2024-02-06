# SAM-SOD


```python train_ada.py adalora --data_path=./data --trset=SCAN --val=scene0616_00```

```python test.py adalora --weight=./weight/adalora/resnet50/base/adalora_resnet50_base_8.pth --data_path=./data --trset=SCAN --vals=SCAN```

Thanks:

+ https://github.com/hitachinsk/SAMed
+ [run files]https://github.com/moothes/SALOD
+ [data]https://drive.google.com/file/d/17X4SiSVuBmqkvQJe_ScVARKPM_vgvCOi/view
+ [loralib]https://github.com/QingruZhang/AdaLoRA