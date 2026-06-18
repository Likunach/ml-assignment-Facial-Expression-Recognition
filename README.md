# Facial Expression Recognition

## კონკურსის მიმოხილვა

Kaggle-ის **Challenges in Representation Learning: Facial Expression Recognition** კონკურსის მიზანია კლასიფიკაცია გავუკეთოთ 48X48-ზე შავ-თეთრი გამოსახულების სურათებს. დეითა მოიცავს შვიდი ემოცციას: Angry (ბრაზი), Disgust (ზიზღი), Fear (შიში), Happy (სიხარული), Sad (სევდა), Surprise (გაკვირვება), Neutral (ნეიტრალური). თითოეული გამოსახულება ინახება pixels სვეტში 2304 მნიშვნელობის სახით (48×48), რომლებიც PyTorch-ში გარდაიქმნება [1, 48, 48].

შეფასების მეტრიკად აღებულია accuracy სატესტო მონაცემებზე, კონკურსი ამჟამად დახურულია, ამიტომ ახალი submission-ის ატვირთვა ვერ მოვახერხე. მიუხედავად ამისა, პროგნოზების ფაილი მაინც მოვამზადე და შეგიძლიათ იხილოთ აქ: submissions/submission.csv (საუკეთესო VGG მოდელის საფუძველზე).

![Sample Predictions](submissions/sample_predictions.png)

---

## პრობლემის გადაჭრის მიდგომა

ამ ამოცანისთვის ავაგე სრული deep learning pipeline: მონაცემთა ჩატვირთვა და აუგმენტაცია → არქიტექტურის იტერაციული განვითარება → ჰიპერპარამეტრების სვიფები (sweeps) → მიზანმიმართული overfit/underfit გაშვებები → შედეგების ანალიზი → submission-ის გენერაცია საუკეთესო მოდელით.

გამოვიყენე 6 CNN არქიტექტურა მზარდი სირთულით: TinyCNN → MediumCNN → DeepCNN → VGGStyleCNN → CustomResNet → TransferResNet18. თითოეულ არქიტექტურას აქვს საკუთარი notebook, სადაც გამოვცადე learning rate, optimizer, augmentation, dropout, batch size, weight decay და scheduler. ყველა ექსპერიმენტი იწერება Weights & Biases-ში group-ის მიხედვით თითო არქიტექტურაზე.

ვარჯიშის დაწყებამდე თითოეული მოდელი გადის forward და backward შემოწმებას: მოწმდება output shape (batch, 7), NaN/inf მნიშვნელობების არარსებობა და ვალიდური გრადიენტები სავარჯიშო (trainable) პარამეტრებზე loss.backward()-ის შემდეგ.

---

## არქიტექტურის დიზაინი

მოთხოვნა იყო რომ დაგვეწყო პატარა მოდელიდან და სირთულე ნელ-ნელა გაგვეზარდა, თითოეული გადაწყვეტილების ახსნით:

| Step | Architecture | What I added | Why |
| ---- | ------------ | ------------ | --- |
| 1 | **TinyCNN** | 2 conv layers, small FC head | სწრაფი baseline pipeline-ის სავალიდაციოდ (მონაცემთა ჩატვირთვა, training loop, Wandb) და LR / optimizer / augmentation-ის გონივრული დიაპაზონების მოსაძებნად ხანგრძლივი ვარჯიშის გარეშე. |
| 2 | **MediumCNN** | 2 more conv blocks  | TinyCNN-ის მაქსიმუმი ~55% val_acc იყო, რაც სავარაუდოდ შეზღუდული ტევადობის გამო underfitting-ით იყო გამოწვეული. მეტი სიღრმის მიცემით უნდა უფრო მეტი შესაძლებლობას ვაძლევთ მოდელს. |
| 3 | **DeepCNN** | BatchNorm + higher dropout | MediumCNN გაუმჯობესდა, მაგრამ მაინც მგრძნობიარეა overfitting-ის მიმართ. BatchNorm ასტაბილურებს ვარჯიშს. dropout + weight decay ამოწმებს, ეხმარება თუ არა რეგულარიზაცია გენერალიზაციას noisy 48×48 გამოსახულებებზე. |
| 4 | **VGGStyleCNN** | Deeper stacked Conv blocks | DeepCNN-მა აჩვენა, რომ სიღრმე + რეგულარიზაცია გამოსადეგიუა. VGG-style ბლოკები ამატებენ მეტ თანმიმდევრულ feature extraction-ს. ეს არის სტანდარტული დიზაინი გამოსახულებების კლასიფიკაციისთვის ResNet-ებამდე. |
| 5 | **CustomResNet** | Skip connections + adaptive pooling | residual connection-ებმა უნდა გააადვილოს ოპტიმიზაცია უფრო ღრმა ქსელებში და შეამციროს დეგრადაცია მეტი ფენის დამატებისას. |
| 6 | **TransferResNet18** | ImageNet pretrained ResNet18 | მაინტერესებდა შეუძლია თუ არა ბუნებრივი გამოსახულებებიდან მიღებულ ზოგად ვიზუალურ მახასიათებლებს, ეჯობნა FER-ზე ნულიდან გაწვრთნილ მოდელისთვის. შევადარე frozen (მხოლოდ head-ის ვარჯიში) vs finetune (მთელი ქსელის ადაპტაცია). |

**მთავარი დასკვნა იტერაციებიდან**: ტევადობაც და რეგულარიზაციაც მნიშვნელოვანია, მაგრამ საუკეთესო მოდელი (VGG) იყო ამ დომენისთვის მორგებული custom CNN და არა transfer learning, რადგან FER2013 გამოსახულებები პატარაა, grayscale-ია და ძალიან განსხვავდება ImageNet-ისგან.

---

## Repository Structure

```
ml-assignment-Facial-Expression-Recognition/
│
├── notebooks/
│   ├── model_experiment_TinyCNN.ipynb           ← basic 2-layer CNN, hyperparameter sweep
│   ├── model_experiment_MediumCNN.ipynb         ← 4-layer CNN
│   ├── model_experiment_DeepCNN.ipynb           ← BatchNorm + Dropout
│   ├── model_experiment_VGG.ipynb               ← VGG-style blocks
│   ├── model_experiment_ResNet.ipynb            ← custom ResNet residual blocks
│   ├── model_experiment_TransferLearning.ipynb  ← pretrained ResNet18 (frozen / finetune)
│   └── model_inference.ipynb                    ← cross-architecture comparison, submission.csv
│
├── src/
│   ├── dataset.py                               ← FERDataset, get_dataloaders, augmentation
│   ├── models.py                                ← all 6 architectures
│   ├── train.py                                 ← shared train loop (Wandb logging, checkpoint)
│   └── utils.py                                 ← Wandb helpers, confusion matrix, early stopping
│
├── checkpoints/                                 ← best weights + results JSON + analysis charts
│   ├── tinycnn_results.json
│   ├── mediumcnn_results.json
│   ├── deepcnn_results.json
│   ├── vgg_results.json
│   ├── resnet_results.json
│   ├── transfer_results.json
│   └── *.png                                    ← class_distribution.png, bar/lr/aug charts, confusion matrices
│
├── data/                                        ← (not in git, too large)
│   ├── train.csv
│   └── test.csv
│
├── submissions/
│   ├── submission.csv                           ← Kaggle-format predictions
│   └── sample_predictions.png
│
├── requirements.txt
└── README.md
```

---

## File Descriptions


| File                                      | Description                                                                                                         |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------  |
| `model_experiment_TinyCNN.ipynb`          | პირველი ბაზისური არქიტექტურა. LR, optimizer, augmentation, dropout, batch size სვიფი. Forward/backward შემოწმება.  |
| `model_experiment_MediumCNN.ipynb`        | უფრო ღრმა 4-conv ბლოკი.                                                                                           |
| `model_experiment_DeepCNN.ipynb`          | BatchNorm + Dropout რეგულარიზაცია.                                                                                 |
| `model_experiment_VGG.ipynb`              | VGG-style ბლოკები.                                                                                                 |
| `model_experiment_ResNet.ipynb`           | Custom ResNet residual connection-ებ + adaptive pooling.                                                            |
| `model_experiment_TransferLearning.ipynb` | ImageNet-ზე წინასწარ გაწვრთნილი ResNet18.                                                                          |               
| `model_inference.ipynb`                   | ადარებს ყველა არქიტექტურას *_results.json-იდან, ირჩევს საუკეთესო მოდელს, აგენერირებს submission.csv-ს.            |
| `src/dataset.py`                          | FERDataset, train/val გაყოფა (80/20), augmentation რეჟიმები: none, light, strong.                                   |
| `src/models.py`                           | ყველა არქიტექტურის განსაზღვრებები.                                                                                 |
| `src/train.py`                            | საზიარო training loop, Adam/SGD, ReduceLROnPlateau, Wandb მეტრიკები, checkpoint-ის შენახვა.                         |
| `submissions/submission.csv`              | მომზადებული submission ფაილი                                                                                     |


---

## მონაცემთა დამუშავება

### ჩატვირთვა და ნორმალიზაცია

- `train.csv` იყოფა **80% train / 20% validation** პროპორციით, კლასების პროპორციები შენარჩუნებულია.
- გამოსახულებები: `pixels` → `48×48` grayscale → `ToTensor()` → `Normalize(mean=0.5, std=0.5)`.

### აუგმენტაცია

გამოვცადე სამი ვარიანტი:


| Mode     | Transforms                                              |
| -------- | ------------------------------------------------------- |
| `none`   | მხოლოდ ToTensor + Normalize                           |
| `light`  | RandomHorizontalFlip, RandomRotation(10°)               |
| `strong` | Flip, Rotation(15°), Affine, ColorJitter, RandomErasing |


აუგმენტაცია ყველაზე მეტად დაეხმარა VGG-სა და ResNet-ში, რადგან საერთოდ აუგმენტაციის არ გამოყენებასთან შედარებით light/strong augmentation ხშირად ამცირებდა overfitting-ს.

### კლასთა დისბალანსი

`train.csv`-ში სულ **28,709** სურათია. კლასები მკვეთრად დისბალანსირებულია `Disgust` ყველაზე ნაკლებადაა წარმოდგენილი, `Happy` ყველაზე მეტად:

![Class Distribution](checkpoints/class_distribution.png)

| Emotion | Count | Share |
| ------- | ----: | ----: |
| Angry | 3,995 | 13.9% |
| Disgust | 436 | **1.5%** |
| Fear | 4,097 | 14.3% |
| Happy | 7,215 | 25.1% |
| Sad | 4,830 | 16.8% |
| Surprise | 3,171 | 11.0% |
| Neutral | 4,965 | 17.3% |

`Disgust` დაახლოებით **16× ნაკლებია**, ვიდრე `Happy`. ამიტომ მხოლოდ accuracy საკმარისი არ არის, confusion matrix-აც ამიტომ დავამატე.

**VGG საუკეთესო მოდელის validation confusion matrix-იდან**:

| True label | Predicted as | Count |
| ---------- | ------------ | ----: |
| Fear | Sad | 216 |
| Neutral | Sad | 184 |
| Sad | Neutral | 179 |
| Angry | Sad | 142 |

ყველაზე დაბალი per-class recall: **Disgust (25.3%)** და **Fear (36.6%)**. ყველაზე მაღალი: **Happy (88.7%)**.

---

## Architecture Comparison (best run per model)


| Architecture        | Best run                                        | val_acc    | # runs       |
| ------------------- | ----------------------------------------------- | ---------- | ------------ |
| **VGG** ✅           | `vgg_adam_0.0003_bs64_strong_do0.25_wd1e-4`     | **0.6546** | 23           |
| ResNet              | `resnet_adam_0.001_bs32_light_do0.5`            | 0.6327     | 24           |
| Transfer (finetune) | `transfer_finetune_adam_1e-4_bs64_strong_sched` | 0.6327     | 19           |
| DeepCNN             | `deep_adam_0.001_bs32_noaug_do0_overfit`        | 0.6243     | 24           |
| MediumCNN           | `medium_adam_0.001_bs64_light_do0.25`           | 0.6169     | 23           |
| TinyCNN             | `tiny_adam_0.001_bs64_light_do0`                | 0.5566     | 16           |
| **Total**           |                                                 |            | **129 runs** |


**საუკეთესო მოდელლი: VGGStyleCNN** `vgg_adam_0.0003_bs64_strong_do0.25_wd1e-4`, val_acc = **65.46%**

ეს მოდელი გამოყენებულ იქნა `submissions/submission.csv`-ის გენერაციისთვის `model_inference.ipynb`-ში.

---

## ვარჯიში და ჰიპერპარამეტრების ოპტიმიზაცია

თითოეული არქიტექტურისთვის გამოვცადე:

- **Learning rate sweep** (adam: 0.01 → 0.0001)
- **Optimizer** (Adam vs SGD)
- **Augmentation** (none / light / strong)
- **Dropout** (0 / 0.25 / 0.5)
- **Batch size** (32 / 64 / 128)
- **Weight decay** (0 / 1e-4 / 5e-4)
- **Scheduler** (ReduceLROnPlateau)
- **Deliberate overfit / underfit** - მოდელის ქცევის გასაგებად

---

### 1. TinyCNN (baseline)

2 convolutional layers + FC head. ~596K parameters.

![TinyCNN All Runs](checkpoints/tinycnn_bar.png)
![TinyCNN LR Sweep](checkpoints/tinycnn_lr.png)
![TinyCNN Augmentation](checkpoints/tinycnn_augmentation.png)
![TinyCNN Dropout](checkpoints/tinycnn_dropout.png)
![TinyCNN Optimizer](checkpoints/tinycnn_optimizer.png)


| run                                                    | val_acc    | assessment                    |
| ------------------------------------------------------ | ---------- | ----------------------------- |
| **best** `tiny_adam_0.001_bs64_light_do0`              | **0.5566** | balanced                      |
| overfit `tiny_adam_0.001_bs32_noaug_do0_overfit`       | 0.4981     | overfit (high train, low val) |
| underfit `tiny_adam_0.0001_bs128_noaug_do0.5_underfit` | 0.4068     | underfit                      |


**დასკვნა***: TinyCNN სწრაფი baseline-ია, მაგრამ შეზღუდული ტევადობა ზღუდავს შედეგს ~55% val_acc-ის ფარგლებში. Light augmentation-ma გააუმჯობესა შედეგი. 

---

### 2. MediumCNN

4 convolutional blocks, ~537K parameters.

![MediumCNN All Runs](checkpoints/mediumcnn_bar.png)
![MediumCNN LR Sweep](checkpoints/mediumcnn_lr.png)
![MediumCNN Augmentation](checkpoints/mediumcnn_augmentation.png)
![MediumCNN Dropout](checkpoints/mediumcnn_dropout.png)
![MediumCNN Overfit](checkpoints/mediumcnn_overfit.png)


| run                                                | val_acc    | assessment         |
| -------------------------------------------------- | ---------- | ------------------ |
| **best** `medium_adam_0.001_bs64_light_do0.25`     | **0.6169** | balanced           |
| overfit `medium_adam_0.001_bs32_noaug_do0_overfit` | 0.5707     | deliberate overfit |
| underfit `medium_adam_0.0001_bs128_noaug_do0.5_underfit` | 0.4056     | deliberate underfit, low LR, high dropout, few epochs |


**Conclusion:** სიღრმის დამატებამ შედეგი ~6%-ით გააუმჯობესა TinyCNN-თან შედარებით. Dropout=0.25 და light augmentation იყო საუკეთესო კომბინაცია. overfit გაშვება (dropout-ის გარეშე, augmentation-ის გარეშე, 40 epoch) აღწევს უფრო მაღალ train accuracy-ს, მაგრამ უფრო დაბალ val accuracy-ს, ვიდრე საუკეთესო დაბალანსებული გაშვება. underfit კონფიგურაცია (lr=0.0001, batch_size=128, dropout=0.5, 5 epoch) განსაზღვრულია model_experiment_MediumCNN.ipynb-ის Supplemental სექციაში და აჩვენებს საპირისპირო ჩავარდნის რეჟიმს: ძალიან მცირე ტევადობა / ძალიან ძლიერი რეგულარიზაცია / ძალიან ცოტა განახლება სასარგებლო მახასიათებლების სასწავლად.


---

### 3. DeepCNN

BatchNorm + Dropout, deeper architecture.

![DeepCNN All Runs](checkpoints/deepcnn_bar.png)
![DeepCNN LR Sweep](checkpoints/deepcnn_lr.png)
![DeepCNN Augmentation](checkpoints/deepcnn_augmentation.png)
![DeepCNN Overfit vs Underfit](checkpoints/deepcnn_overfit_underfit.png)
![DeepCNN Confusion Matrix](checkpoints/deepcnn_confusion_matrix.png)


| run                                                    | val_acc    | assessment                |
| ------------------------------------------------------ | ---------- | ------------------------- |
| **best** `deep_adam_0.001_bs32_noaug_do0_overfit`      | **0.6243** | overfit run, but best val |
| underfit `deep_adam_0.0001_bs128_noaug_do0.5_underfit` | 0.4796     | underfit                  |


**დასკვნა**: DeepCNN-მა მიაღწია ~62.4% val_acc-ს. overfit/underfit ჩარტი გვიჩვენებს ორივე მხარეს, underfit გაშვება ძალიან დაბალია (~48%), ხოლო overfit გაშვება ინარჩუნებს ვარგის val accuracy-ს მაღალი train accuracy-ის მიუხედავად.

---

### 4. VGGStyleCNN (best model)

VGG-style blocks with repeated Conv+ReLU+MaxPool.

![VGG All Runs](checkpoints/vgg_bar.png)
![VGG LR Sweep](checkpoints/vgg_lr.png)
![VGG Augmentation](checkpoints/vgg_augmentation.png)
![VGG Dropout](checkpoints/vgg_dropout.png)
![VGG Overfit vs Underfit](checkpoints/vgg_overfit_underfit.png)
![VGG Confusion Matrix](checkpoints/vgg_confusion_matrix.png)


| run                                                   | val_acc    | assessment |
| ----------------------------------------------------- | ---------- | ---------- |
| **best** `vgg_adam_0.0003_bs64_strong_do0.25_wd1e-4`  | **0.6546** | balanced  |
| overfit `vgg_adam_0.001_bs32_noaug_do0_overfit`       | 0.5378     | overfit    |
| underfit `vgg_adam_0.0001_bs128_noaug_do0.5_underfit` | 0.4894     | underfit   |


**დასკვნა:** VGG-მა საუკეთესო შედეგი აჩვენა custom CNN-ებს შორის. ოპტიმალური კომბინაცია: `lr=0.0003`, `strong` augmentation, `dropout=0.25`, `weight_decay=1e-4`. Confusion matrix-ში Happy-ს recall 88.7%ა, Disgust-ს კი მხოლოდ 25.3%, ხშირია შეცდომები Fear→Sad და Neutral→Sad.

---

### 5. CustomResNet

Residual blocks + adaptive average pooling.

![ResNet All Runs](checkpoints/resnet_bar.png)
![ResNet LR Sweep](checkpoints/resnet_lr.png)
![ResNet Augmentation](checkpoints/resnet_augmentation.png)
![ResNet Overfit vs Underfit](checkpoints/resnet_overfit_underfit.png)
![ResNet Confusion Matrix](checkpoints/resnet_confusion_matrix.png)


| run                                                      | val_acc    | assessment |
| -------------------------------------------------------- | ---------- | ---------- |
| **best** `resnet_adam_0.001_bs32_light_do0.5`            | **0.6327** | balanced   |
| overfit `resnet_adam_0.001_bs32_noaug_do0_overfit`       | 0.6076     | overfit    |
| underfit `resnet_adam_0.0001_bs128_noaug_do0.5_underfit` | 0.3572     | underfit   |


**დასკვნა**: ResNet-მა მიაღწია ~63.3% val_acc-ს. Skip connection-ები დაეხმარა სტაბილურ ვარჯიშს, მაგრამ ამ ზომის მონაცემთა ნაკრებზე VGG-ს ვერ აჯობა.


---

### 6. TransferResNet18 (Pretrained)

ImageNet pretrained ResNet18, 1-channel input is repeated to 3 channels. Two modes: `frozen` (FC head only) and `finetune` (full network).

![Transfer All Runs](checkpoints/transfer_bar.png)
![Transfer Frozen vs Finetune](checkpoints/transfer_frozen_vs_finetune.png)
![Transfer LR Finetune](checkpoints/transfer_lr_finetune.png)
![Transfer Overfit vs Underfit](checkpoints/transfer_overfit_underfit.png)
![Transfer Confusion Matrix](checkpoints/transfer_confusion_matrix.png)


| run                                                         | val_acc    | assessment                  |
| ----------------------------------------------------------- | ---------- | --------------------------- |
| **best** `transfer_finetune_adam_1e-4_bs64_strong_sched`    | **0.6327** | finetune                  |
| overfit `transfer_finetune_adam_1e-3_bs32_noaug_overfit`    | 0.6109     | overfit                     |
| underfit `transfer_frozen_adam_0.0001_bs128_noaug_underfit` | 0.2698     | underfit (frozen, 3 epochs) |


დასკვნა: Finetune რეჟიმი მნიშვნელოვნად უკეთესია frozen-ზე (~63.3% vs ~35% საუკეთესო frozen). Transfer learning-მა ვერ აჯობა VGG-ს, რადგან ImageNet-ის წინასწარი წვრთნა ეფუძნება დიდ RGB ბუნებრივ გამოსახულებებს, ხოლო FER2013 შეიცავს პატარა 48×48 grayscale სახის ნაჭრებს.

---

## Overfitting / Underfitting ანალიზი

ყველა არქიტექტურაზე გავუშვი verfit და underfit კონფიგურაციები. ყველა გაშვებაზე Wandb-ში ლოგდება `train_acc` და `val_acc` თითოეულ ეპოქაზე. train–val დაშორება overfitting-ის პირდაპირი ნიშანია.

### კონფიგურაციების შაბლონი

| რეჟიმი | ტიპური პარამეტრები | მიზანი |
|--------|-------------------|--------|
| **Overfit** | `lr=0.001`, `bs=32`, `aug=none`, `dropout=0`, `weight_decay=0`, 30–40 epoch | მაღალი ტევადობა + სუსტი რეგულარიზაცია → მოდელი ისწავლის train-ს, val ჩამორჩება |
| **Underfit** | `lr=0.0001`, `bs=128`, `aug=none`, `dropout=0.5`, 5 epoch (Transfer frozen: 3 epoch) | ძალიან ნელი სწავლა + ძლიერი რეგულარიზაცია + ცოტა epoch → მოდელი ვერ იჭერებს რთულ ნიმუშებს |
| **Balanced** | sweep-იდან შერჩეული aug / dropout / wd / scheduler | საუკეთესო კომპრომისი train და val შორის |

### შედეგები არქიტექტურების მიხედვით

| Architecture | Best val_acc | Overfit val_acc | Underfit val_acc | What caused it |
|--------------|-------------:|----------------:|-----------------:|----------------|
| TinyCNN | 0.5566 | 0.4981 | 0.4068 | Small capacity. Even on overfit runs train is much higher than val. On underfit the model cannot learn well. |
| MediumCNN | 0.6169 | 0.5707 | 0.4056 | More depth raised overfitting risk. Augmentation and dropout improved generalization. |
| DeepCNN | 0.6243* | 0.6243* | 0.4796 | The overfit setup gave the best val score. Underfit at about 48% means the model learns too little in practice. |
| VGG | 0.6546 | 0.5378 | 0.4894 | The largest model falls hard on val when overfitting. On the balanced run strong augmentation and weight decay keep train and val closer. |
| ResNet | 0.6327 | 0.6076 | 0.3572 | Skip connections stabilize training but the overfit setup still shows a train val gap. Underfit is the weakest of all runs. |
| Transfer | 0.6327 | 0.6109 | 0.2698 | Overfit came from a high learning rate during finetune. Underfit used a frozen backbone so only the head trained and the signal stayed weak. |

\*DeepCNN-ის საუკეთესო გაშვება თავად overfit კონფიგურაციაა ეს აჩვენებს იმას, რომ ამ არქიტექტურაზე რეგულარიზაციის გარეშე მაღალი val მიღწევადია, მაგრამ train–val gap Wandb-ზე მაინც ჩანს.

### რა ვნახე Wandb-ში (train vs val)

- **Overfit გაშვებები:** `train_acc` სწრაფად იზრდება, `val_acc` ჭეშმარიტება ან ჩამორჩება - კლასიკური overfitting.
- **Underfit გაშვებები:** train და val ორივე დაბალია და ნელი იზრდება - მოდელს არ აქვს საკმარისი სიმძლავრე ან დრო სასარგებლო ნიმუშების სასწავლად.
- **Balanced best runs:** train და val ერთად იზრდება, gap პატარა რჩება - აქედან ავირჩიე submission-ისთვის VGG.

### დასკვნა (overfit / underfit)

1. **Capacity** (Tiny → VGG) ზრდის overfitting-ის რისკს, თუ aug/dropout/wd არ არის.
2. **Augmentation + dropout + weight decay** ამცირებს train–val gap-ს და აუმჯობესებს გენერალიზაციას.
3. **Underfit** განზრახ გაშვებებმა დაადასტურა, რომ პრობლემა ზოგჯერ არა overfitting-შია, არამედ ძალიან ნელი სწავლაში ან ძალიან ძლიერ რეგულარიზაციაში.
4. **Transfer frozen underfit** განსხვავებული შემთხვევაა: backbone ფიქსირებულია, ამიტომ მოდელი ვერ ადაპტირდება FER-ის პატარა grayscale სახეებზე.

ჩარტები: `*_overfit_underfit.png` (Deep, VGG, ResNet, Transfer), `mediumcnn_overfit.png`. სრული train/val კურვები - [Wandb პროექტი](https://wandb.ai/lchit22-free-university-of-tbilisi-/fer2013-expression-recognition) და [Wandb Report](https://api.wandb.ai/links/lchit22-free-university-of-tbilisi-/rsznxffa).

---

## Forward და Backward შემოწმება

თითოეულ საექსპერიმენტო notebook-ს აქვს სექცია **3b sanity check**:

1. **Forward check** -  გადაეცემა რეალური batch, მოწმდება output shape (batch, 7), NaN/inf.
2. **Backward check** - `CrossEntropyLoss` + `loss.backward()`, სავარჯიშო პარამეტრებს აქვთ არანულოვანი, ვალიდური გრადიენტები.

---

## საუკეთესო მოდელის შერჩევა და Submission

model_inference.ipynb ტვირთავს checkpoints/*_results.json-ს, ადარებს თითოეული არქიტექტურის საუკეთესო გაშვებას და ირჩევს გლობალურ გამარჯვებულს:

```python
best = max(all_arch_results, key=lambda x: x['best_val_acc'])
# → vgg - vgg_adam_0.0003_bs64_strong_do0.25_wd1e-4 - val_acc=0.6546
```

`val_acc` გამოვიყენე როგორც მთავარი შერჩევის მეტრიკა, რადგან:

- ის უშუალოდ ასახავს შესრულებას სავალიდაციო ნაკრებზე.
- Train/val სხვაობა მონიტორინგდებოდა overfitting-ის გამოსავლენად.
- Kaggle-ის მეტრიკა არის accuracy, val_acc მისი ყველაზე ახლო ანალოგია.

**Submission**: submissions/submission.csv მომზადდა საუკეთესო VGG checkpoint-ის გამოყენებით. კონკურსი დახურულია, ამიტომ ონლაინ ქულის მიღება ვერ მოხერხდა.

---

## Wandb ექსპერიმენტები

ყველა ექსპერიმენტი ასახულია აქ: [wandb.ai/lchit22-free-university-of-tbilisi-/fer2013-expression-recognition](https://wandb.ai/lchit22-free-university-of-tbilisi-/fer2013-expression-recognition)

თითოეული გაშვება აღწერს:

**პარამეტრები:** `lr`, `optimizer`, `batch_size`, `aug`, `dropout`, `weight_decay`, `scheduler`, `epochs`

**მეტრიკები (თითოეულ ეპოქაზე):**

- `train_loss`, `val_loss`
- `train_acc`, `val_acc`
- `lr`


### ექსპერიმენტის სტრუქტურა


| Wandb group | # runs  | parameters explored                                                                |
| ----------- | ------- | ---------------------------------------------------------------------------------- |
| `tiny`      | 16      | lr, optimizer, aug, dropout, batch_size, overfit/underfit, EDA                     |
| `medium`    | 23      | lr, optimizer, aug, dropout, batch_size, weight_decay, scheduler, overfit/underfit |
| `deep`      | 24      | lr, optimizer, aug, dropout, batch_size, weight_decay, scheduler, overfit/underfit |
| `vgg`       | 23      | lr, optimizer, aug, dropout, batch_size, weight_decay, overfit/underfit            |
| `resnet`    | 24      | lr, optimizer, aug, dropout, batch_size, weight_decay, overfit/underfit            |
| `transfer`  | 19      | lr, optimizer, aug, mode (frozen/finetune), scheduler, overfit/underfit            |
| **Total**   | **129** |                                                                                    |


---

## Wandb Report (bonus)

შერჩეული არქიტექტურათაშორისი ანალიზი, ჩარტები და გაშვებების შედარებები:

**[Wandb Report — Facial Expression Recognition](https://api.wandb.ai/links/lchit22-free-university-of-tbilisi-/rsznxffa)**

---

## გამოცდილება და დასკვნები

ამ პროექტმა აჩვენა, რომ FER2013-ზე კარგი შედეგი დამოკიდებულია უფრო მეტ რამეზე, ვიდრე უბრალოდ უფრო დიდი მოდელის არჩევაა, augmentation, რეგულარიზაცია და სწორი learning rate კრიტიკულია.

ძირითადი დასკვნები:

არქიტექტურის იტერაციული დიზაინი (Tiny → Medium → Deep → VGG → ResNet → Transfer) ეხმარება სისტემურად დადგინდეს, არის თუ არა პრობლემა underfitting თუ overfitting-ში.
Augmentation ბევრს ნიშნავს დაბალი გარჩევადობის 48×48 გამოსახულებებზე.
Forward/backward შემოწმება აუცილებელია ვარჯიშამდე - განსაკუთრებით transfer learning-ში, სადაც frozen პარამეტრებმა გრადიენტები არ უნდა მიიღონ.
Wandb tracking-მა შესაძლებელი გახადა ოპტიმალური ჰიპერპარამეტრების მოძებნა 129 გაშვებაში.
წინასწარ გაწვრთნილმა ResNet18-მა ვერ აჯობა VGG-ს, თუმცა სასარგებლო იყო: ImageNet-ის წონები მიმართულია დიდ RGB სცენებზე და არა პატარა grayscale სახის ნაჭრებზე, ამიტომ end-to-end VGG ამ ვარიანტისთვის უკეთესი არჩევანი აღმოჩნდა.

---

## როგორ გავუშვათ

```bash
pip install -r requirements.txt
pip install jupyter   # optional - for running .ipynb notebooks
```

**GPU (optional):** For NVIDIA GPUs (e.g. RTX 50-series), install CUDA-enabled PyTorch if the default wheel is CPU-only:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

1. ჩამოტვირთეთ FER2013 მონაცემები data/train.csv და data/test.csv-ში.
2. გაუშვით setup უჯრა ნებისმიერ model_experiment_*.ipynb notebook-ში.
3. ნახეთ ექსპერიმენტის შედეგები Wandb-ზე ან checkpoints/*_results.json-ში.
4. გაუშვით model_inference.ipynb საბოლოო შედარებისა და submission-ის გენერაციისთვის.
