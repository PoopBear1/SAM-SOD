# SAM-SOD


```python train_ada.py adalora --data_path=./data --trset=SCAN --val=scene0616_00```

```python test.py adalora --weight=./weight/adalora/resnet50/base/adalora_resnet50_base_8.pth --data_path=./data --trset=SCAN --vals=SCAN```



适配LoRA之后，对于室内场景微调的想法：
* 对于每张图片利用已知的类别分别生成binary mask,然后分别分割单独计算损失.
  * Q:
    * 相当于一张图片给了n_class个结果, 这样出来的结果会是正确的吗
* 修改最后的输出层，即MaskDecoder更改为已知的n_class.
  * Q:
    * 最后一层需要单独训练，冻结之前的权重只训练最后一层合适吗



Thanks:

+ https://github.com/hitachinsk/SAMed
+ [run files]https://github.com/moothes/SALOD
+ [data]https://drive.google.com/file/d/17X4SiSVuBmqkvQJe_ScVARKPM_vgvCOi/view
+ [loralib]https://github.com/QingruZhang/AdaLoRA