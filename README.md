# MILTestTask - OCR

Мы предлагаем обучить модель для решения задачи Layout Detection на нашем [датасете](https://drive.google.com/file/d/1euOGyo8jzP-iJF_WMuwTtBzrRsvQ4h3c/view?usp=sharing).  
В архиве есть папка `data` с изображениями, и 2 json файла в [формате COCO для задачи detection](https://cocodataset.org/#format-data), c train и test частями соответственно. 
Данные для сегментации приведены в формате полигонов.
  
Для работы с форматом COCO рекомендуется использовать библиотеку `pycocotools`.

Код необходимый для получения результатов обучения модели нужно приложить в форке этого репозитория.  
Отчет по процессу решения и итоговым результатам желательно оформить как jupyter ноутбук с метриками mean IoU для тестовой и трейновой частей.

Перед решением рекомендуется взглянуть на датасет. 
В качестве базовых решений предлагаем ознакомиться со статьями [раз](https://arxiv.org/pdf/1512.02325.pdf) и [два](https://link.springer.com/chapter/10.1007/978-3-319-95957-3_30).


# Solution ​

## Preprocessing

### COCO format to mask

```python
def polygon_to_mask(xy, w, h, c, normed_dists=True):
 ​xy = np.array(xy)
 ​xy = np.array(xy[0])
 ​if normed_dists:
   ​xy[::2] *= w
   ​xy[1::2] *= h
 ​xy = xy.reshape(-1, 2)
 ​xy = [tuple(a) for a in xy]
 ​img = Image.new('L', (w, h), 0)
 ​if len(xy) < 2:
   ​mask = np.array(img)
   ​return mask

 ​ImageDraw.Draw(img).polygon(xy, outline=c, fill=c)
 ​mask = np.array(img)
 ​return mask


def get_mask(image, annots, normed_dists=True):
 ​w, h = image['width'], image['height']
 ​mask = np.zeros((h, w))
 ​for a in annots:
   ​data = polygon_to_mask(a['segmentation'], w, h, a['category_id'], normed_dists)
   ​mask = np.where(data, a['category_id'], mask)
 ​return mask

```

All masks were built with PIL library. At first all segmentations were transformed from polygon to each mask with ```polygon_to_mask``` function, then all masks were merged together with ```get_mask``` function. In train we've got normed widths and heights, in test - not normed.

### Image processing

#### Augmentations
Augmentations didn't helped to train model.

#### Other transforms
Resizing to 256, 256

Normalization to ```mean=[0.485, 0.456, 0.406],
                                ​std=[0.229, 0.224, 0.225]```



## Models

I used models from [segmentation models pytorch](https://github.com/qubvel/segmentation_models.pytorch) library

### Tested decoders
1) Unet
2) FPN
3) DeepLabV3
4) DeepLabV3+

All decoders were tested on **EfficientNetB0** ImageNet pretrained encoder

After choosing best decoder (DeepLabV3+), I used several EfficientNet encoders

Best encoder was **EfficientNetB7**

## Results

| Train mIOU | Test mIOU |
| --- | --- |
| 98.3761% | 95.6109% |


I got 95.61% mean IOU on test dataset with best model and best augmentations combination (no augmentations