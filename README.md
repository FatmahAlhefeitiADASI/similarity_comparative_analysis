# Comprehensive Similarity Comparative Tables
## Description
A comprehensive similarity comparative tables has been done for different PyTorch models and per-trained weights. The tables separated by GFLOPS ranges starting from 0 up to 15. For each model, the positive and negative similarities have been calculated separately in (%) and reported with its embedding inference time (ms), similarity inference time (ms).

In addition to the dimensionality of the embedded vectors, it was reported for each model's vectors.

Mentioning the GFLOPS which is measure the computer’s processing speed in terms of floating-point operations (involving mathematical operations with numbers that have decimal points). It is commonly used to quantify the performance of GPUs in term of how many billion floating-point operations it can perform per second. Acc1 and acc5 represents the proportion of the test samples for which the correct label is the model’s highest-confidence prediction of top-one and top-five accuracy. 




## Device used
NVIDIA Jetson AGX Orin (utilizing GPU)



## Input Data
input data differ depends on the similarity type. 
- For **positive similarity**, anchor and source image were used (similar). 
- For **negative similarity**, anchor and negative image were used (dissimilar).



## Transform Function
**‘torchvision.transforms’** module within PyTorch’s torchvision library was used. It provides set of common image transformations for image preprocessing and augmentation. Including the basic operations and converting images to tensors. 

**‘Compose’** class allows to compose multiple image transformations into a single transformation pipeline (sequentially) to input image.

```
transform = transforms.Compose([  
    transforms.Resize(232), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```
**Resize, CentreCrop, and Normalize** parameters differ based on the model (note: the above example is for ‘regnet_x_16gf’ model).




## Similarity Algorithm
**Cosine similarity** was used between two vectors represented by PyTorch tensors. Tensors embeddings moved from GPU to CPU and then converted to numpy arrays to perform the similarity caalculation using the following equation:
![Cosine Equation](/home/adasi/git-sim/cosine_eq.png). And returns the calculated cosine similarity value.

```
def cosine_similarity(emb1, emb2):
    if emb1.is_cuda:
        emb1 = emb1.cpu()
    if emb2.is_cuda:
        emb2 = emb2.cpu()
    emb1_np = emb1.numpy()
    emb2_np = emb2.numpy()
    return np.dot(emb1_np, emb2_np.T) / (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np))
```



## Termonology
- **NN Archeticture**: Neural Network Archeticture (model)
- **(a)**: anchor image/template
- **(s)**: source image (positive)
- **(n)**: negative image
- **(n1)**: rezised negative image
- **GFLOPS)**: Giga Floating Point Operations Per Second
  

## Note
- (a) image resolution =  (size = 9.7 kB)
  * Width: 301 pixels  
  * Height: 167 pixels
- (s) image resolution = (size = 10.7 kB)
  * Width: 259 pixels  
  * Height: 194 pixels
- (n) image resolution = (size = 236.2 kB)
  * Width: 2022 pixels  
  * Height: 1349 pixels
- (n1) image resolution: (size = 9.7 kB)
  * Width: 300 pixels  
  * Height: 170 pixels
- Negative similarity for GFLOPS ranges from 0-3 was calculated for image (n) and (n1) **separetly**.
- Negative similarity for GFLOPS ranges from 4-8 was calculated for image (n1) **only**.


## Comparative Tables


<details>  
<summary>1. GFLOPS Range (0)</summary>
    
|**NN Architecture**|alexnet|efficientnet_b0|efficientnet_b1(v2)|mnasnet0_5|mnasnet0_75|mnasnet1_0|mnasnet1_3|mobilenet_v2 (v2)|mobilenet_v3_large(v2)|mobilenet_v3_small|regnet_x_400mf (v2)|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Positive Similarity**|-|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**| 22.9452|76.0708|106.3330|53.2820|55.9785|58.9318|64.6741|63.5908|62.2003|57.5352|74.1813|
| **Similarity Inference Time(ms)***|1.0910|1.9257|0.9677|1.7705|1.5173|1.9150|2.6190|1.6997|1.3955|1.5497|1.8296|
| **Anchor & Positive Similarity**|54.0608|58.7266|62.6513|24.7069|34.0333|41.4679|30.7476|53.7780|81.3388|79.1964|85.3203|
| **Negative Similarity (n)**| -|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|66.6418|123.0218|135.6370|91.1896|92.5117|92.3293|92.2983|91.9986|102.9584|90.6320|112.9642|
| **Similarity Inference Time**| 1.7035|0.9840|1.3869|1.9929|1.0848|1.1435|1.0560|0.8957|0.9561|1.1070|1.0641|
| **Anchor & Negative Similarity**| 20.9051|-1.8293|-3.1438|1.4024|4.1157|14.5433|4.1319|25.2240|40.3210|35.2345|46.9519|
| **Negative Similarity (n1)-resized**| -|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|23.1783|85.9363|104.8846|55.4149|56.0505|55.1937|53.7698|61.7499|67.3475|65.4585|78.7289|
| **Similarity Inference Time**| 1.1210|0.8719|0.8240|1.5688|1.6725|1.8065|1.9388|1.4744|1.0903|1.9948|1.3828|
| **Anchor & Negative Similarity**| 21.2423|0.6875|0.5179|1.2124|6.0947|14.5142|4.8405|26.2946|39.1919|30.6165|49.9285|
| **Embedded Vectors Dimension**| 9216|1280|1280|62720|62720|62720|62720|62720|960|576|400|
| **GFLOPS (Giga Floating Point Operations Per Second)**| 0.71|0.39|0.69|0.1|0.21|0.31|0.53|0.3|0.22|0.06|0.41|
| **acc1**| 56.522|77.692|79.838|67.734|71.18|73.456|76.506|72.154|75.274|67.668|74.864|
| **acc5**|79.066|93.532|94.934|87.49|90.496|91.51|93.522|90.822|92.566|87.402|92.322|
### **Analysis:** 
- **alexnet** has the smallest embedding inference time (positive similarity)
- **efficientnet_b0** & **efficientnet_b1 (v2)** has negative values in the anchor & negative similarity (negativr similarity) of **image (n)**
- **efficientnet_b1(v2)** has the smallest similarity inference time (negative similarity)
- **mobilenet_v3_large(v2)** & **regnet_x_400mf (v2)** models reach +80% in anchor & positive similarity (positive similarity), while **mobilenet_v3_small** almost 80%
- **regnet_x_400mf(v2)** has the smallest vector dimentionaloty = 400
</details>



<details>  
<summary>1.1 GFLOPS Range (0)</summary>

|**NN Architecture**|regnet_x_800mf(v2)|regnet_y_400mf(v2)|regnet_y_800mf(v2)|shufflenet_v2_x0_5|shufflenet_v2_x1_0|shufflenet_v2_x1_5|shufflenet_v2_x2_0|squeezenet1_0|squeezenet1_1|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Positive Similarity**|-|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|60.6577|88.3641|78.8457|62.2530|63.0331|59.3038|65.4593|38.1634|38.6057|
| **Similarity Inference Time(ms)**|1.8914|1.6398|1.0743|1.6510|1.5068|1.5833|2.1164|1.5955|1.6332|
| **Anchor & Positive Similarity**|86.6694|86.3151|80.5388|24.0615|30.6895|29.0416|29.2019|23.7270|25.2612|
| **Negative Similarity (n)**| -|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|97.8277|120.5173|115.4060|98.5503|97.7421|97.1923|93.2729|77.8642|75.0432|
| **Similarity Inference Time**|0.9913|0.9992|0.9663|1.1430|1.5476|0.9713|1.0717|1.8344|1.5481|
| **Anchor & Negative Similarity**|47.4506|47.4931|43.7597|6.0648|8.9503|7.4715|1.8353|6.1105|5.8580|
| **Negative Similarity (n1)-resized**| -|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|65.7325|87.4243|80.8372|64.4193|69.9468|66.6306|74.7674|34.9214|41.7392|
| **Similarity Inference Time**|1.7939|1.8966|1.0331|1.8563|1.7314|1.8320|1.8895|1.5886|1.8382|
| **Anchor & Negative Similarity**|50.3918|49.3711|43.2646|6.7157|9.3395|8.4113|1.9839|5.4558|5.1805|
| **Embedded Vectors Dimension**|672|440|784|50176|50176|50176|100352|86528|86528|
| **GFLOPS (Giga Floating Point Operations Per Second)**|0.80|0.40|0.83|0.04|0.14|0.30|0.58|0.82|0.35|
| **acc1**|77.522|75.804|78.828|60.552|69.362|72.996|76.23|58.092|58.178|
| **acc5**|93.826|92.742|94.502|81.746|88.316|91.086|93.006|80.42|80.624|
### **Analysis:** 

</details>


<details>  
<summary>2. GFLOPS Range (1-3)</summary>
    
|**NN Architecture**|densenet121|densenet169|efficientnet_b2|efficientnet_b3|googlenet|regnet_x_1_6gf(v2)|regnet_x_3_2gf(V2)|resnet18|resnet34|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Positive Similarity**|-|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|129.3135|164.8493|100.2641|108.4678|74.7306|70.9918|96.2665|35.5673|50.2386|
| **Similarity Inference Time(ms)***|1.8938|1.5225|1.8981|0.8507|1.6232|1.9624|0.8409|1.4052|1.4789|
| **Anchor & Positive Similarity**|46.7121|44.4098|60.4621|70.5458|83.2081|87.6996|85.1251|89.4667|85.0670|
| **Negative Similarity (n)**| -|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|155.3297|205.0416|136.4136|148.7305|105.8691|105.6259|121.8581|80.9047|88.0032|
| **Similarity Inference Time**|1.4734|1.5128|1.9474|1.3773|0.9210|1.6038|1.0743|1.1153|1.7700|
| **Anchor & Negative Similarity**|4.5747|5.0117|-4.1283|-14.1572|48.4169|48.5893|44.0161|50.9895|50.3405|
| **Negative Similarity (n1)-resized**| -|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|130.9700|169.5974|101.8236|108.9261|74.1434|68.3005|95.3572|37.7469|56.3161|
| **Similarity Inference Time**|1.5304|1.6310|0.8900|1.2493|1.0204|1.7436|1.5919|1.4832|1.5001|
| **Anchor & Negative Similarity**|5.2418|5.6970|-5.7051|-12.9239|50.0063|53.1077|47.1533|56.8162|51.2528|
| **Embedded Vectors Dimension**|50176|81536|1408|1536|1024|912|1008|512|512|
| **GFLOPS (Giga Floating Point Operations Per Second)**|2.83|3.36|1.09|1.83|1.50|1.60|3.18|1.81|3.66|
| **acc1**|74.434|75.6|80.608|82.008|69.778|79.668|81.196|69.758|73.314|
| **acc5**|91.972|92.806|95.31|96.054|89.53|94.922|95.43|89.078|91.42|
### **Analysis:** 

</details>




<details>  
<summary>3. GFLOPS Range (4-6)</summary>
    
|**NN Architecture**|convnext_tiny|densenet201|efficientnet_b4|resnext50_32x4d(v2)|resnet50(v2)|resnet50(v1)|swin_t|swin_v2_t|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Positive Similarity**|-|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|57.4660|190.1944|140.6250|64.2722|64.6493|65.7146|96.7839|119.6494|
| **Similarity Inference Time(ms)***|1.3921|1.5774|1.4448|1.6246|1.3900|1.5535|1.6072|1.8780|
| **Anchor & Positive Similarity**|81.8123|42.9261|71.3894|82.1966|90.5058|90.6283|76.9918|65.1037|
| **Negative Similarity (n1)-resized**| -|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|62.8245|193.9106|131.2571|73.2257|68.2092|67.1537|104.7289|123.0156|
| **Similarity Inference Time**|1.8282|1.0900|1.5678|1.8406|1.4532|1.7629|1.7266|1.7190|
| **Anchor & Negative Similarity**|36.6136|6.5070|30.6059|47.3308|51.9603|54.9102|4.4156|7.1858|
| **Embedded Vectors Dimension**|768|94080|1792|2048|2048|2048|768|768|
| **GFLOPS (Giga Floating Point Operations Per Second)**|4.46|4.29|4.39|5.71|4.09|4.09|4.49|5.94|
| **acc1**|82.52|76.896|83.384|81.198|80.858|76.13|81.474|82.072|
| **acc5**|96.146|93.37|96.594|95.34|95.434|92.862|95.776|96.132|
### **Analysis:** 

</details>


<details>  
<summary>4. GFLOPS Range (7-8)</summary>
    
|**NN Architecture**|convnext_small|densenet161|efficientnet_v2_s|regnet_y_8gf(v2)|resnet101(v2)|swin_s|vgg11_bn|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Positive Similarity**|-|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|89.8159|168.3393|149.7233|107.0871|115.3145|151.2792|30.0901|
| **Similarity Inference Time(ms)***|1.4477|3.3009|1.7450|2.6221|4.7281|1.6119|7.9021|
| **Anchor & Positive Similarity**|79.2268|42.9442|80.8309|87.1618|84.5940|75.8689|38.3932|
| **Negative Similarity (n1)-resized**| -|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|97.8384|172.3936|151.3839|114.8224|118.9411|160.2235|36.5844|
| **Similarity Inference Time**|1.2455|2.3272|1.8921|3.6888|3.7596|1.6117|8.2054|
| **Anchor & Negative Similarity**|49.6226|6.6445|-0.0798|51.3794|51.1027|5.0749|14.1395|
| **Embedded Vectors Dimension**|768|108192|1280|2016|2048|768|25088|
| **GFLOPS (Giga Floating Point Operations Per Second)**|8.68|7.73|8.37|8.47|7.80|8.74|7.61|
| **acc1**|83.616|77.138|84.228|82.828|81.886|83.196|70.37|
| **acc5**|96.65|93.56|96.878|96.33|95.78|96.36|89.81|
### **Analysis:** 

</details>


<details>  
<summary>5. GFLOPS Range (9-ongoing)</summary>
    
|**NN Architecture**|vgg16 (v1)|efficientnet_b5|wide_resnet50_2 (v2)|swin_v2_s|regnet_x_16gf|
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Positive Similarity**|-|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|49.5183|190.2869|67.6265|214.0138|97.6832|
| **Similarity Inference Time(ms)***|13.1276|77.7090|9.6200|1.6263|1.4076|
| **Anchor & Positive Similarity**|38.7325|65.5254|81.8282|76.0956|85.6198|
| **Negative Similarity (n1)-resized**| -|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|59.5284|205.9166|76.2448|216.3146|87.3930|
| **Similarity Inference Time**|26.8877|108.0291|17.1156|3.4590|2.9933|
| **Anchor & Negative Similarity**|11.5088|6.3949|34.0802|4.7213|53.6572|
| **Embedded Vectors Dimension**|25088|2048|2048|768|2048|
| **GFLOPS (Giga Floating Point Operations Per Second)**|15.47|10.27|11.40|11.55|15.94|
| **acc1**|71.592|83.444|81.602|83.712|80.058|
| **acc5**|90.382|96.628|95.758|96.816|94.944|
### **Analysis:** 

</details>










