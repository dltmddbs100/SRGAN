# SRGAN with pytorch
   
It's based on [Photo-Relistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802).  
You can find the paper review in my [post](https://deepkerry.tistory.com/20?category=1189783).

<br/>

### **Instruction**

<img src = "https://user-images.githubusercontent.com/55730591/146665847-b8883e93-71d8-4938-a12d-668bdcc96620.jpg" width="80%" height="80%">

+ skip-connection과 residual blocks를 이용한 design
+ pretrained된 VGG19 network의 feature map의 euclidean distance를 content loss로 사용
+ srgan loss = adversarial loss + content loss + TVloss
+ 구현시 anisotropic total variation loss인 TV loss를 도입해 적용
+ metric으로 mse와 psnr을 도출

<br/>

### **Dataset & Args**

<p align="center"><img src = "https://user-images.githubusercontent.com/55730591/146666341-bf87c09d-3cb3-488e-9f4b-6fa891f331e8.png"></p>

+ 카메라로 촬영된 실제 이미지 약 700여장을 사용 (https://drive.google.com/file/d/1VIcrddpM_Inosk0wVQZajl59YeFzNj5z/view)
+ 원본 size가 (2448, 3264) 및 (1224,1632)로 고용량
+ GPU 상의 한계로 원본 데이터를 Resize(BICUBIC interpolation)를 통해 low resolution image와 high resolution image를 구성
+ 각각 (153,204) / (306,408) 의 크기로 구성
+ 4배까지 upscale이 가능, 2배 upscale 적용
+ 각 epoch마다 testset의 mse, psnr값과 output 이미지를 반환하도록 구성

<br/>

### **Results**

+ Epochs의 증가에 따라 Discriminator의 loss는 감소하지않고 고정적인 모습을 보이며, Generator loss는 지속적으로 감소
+ 동시에 testset의 psnr score가 점차 개선

<img src = "https://user-images.githubusercontent.com/55730591/146666118-d3be6eca-7cfc-44f5-b388-32ef5d008725.png" width="80%" height="80%">


✅ **TEST SET RESULTS**
| Epochs | MSE | PSNR | 
| :---------: | :----------: | :----------: |
| 1 | 0.008219 | 26.979892 |
| 10 | 0.002429 | 32.254258 |
| 20 | 0.002084 | 32.913233 |
| 30 | 0.002235 | 32.607018 |
| 40 | 0.001854 | 33.420432 |
| 50 | 0.001854 | 33.428787 |
