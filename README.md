# Comprehensive Similarity Comparative Tables
## Description
A comprehensive similarity comparative tables has been done for different PyTorch models and per-trained weights. The tables separated by GFLOPS ranges starting from 0 up to 15. For each model, the positive and negative similarities have been calculated separately in (%) and reported with its embedding inference time (ms), similarity inference time (ms).

Mentioning the GFLOPS which is measure the computer’s processing speed in terms of floating-point operations (involving mathematical operations with numbers that have decimal points). It is commonly used to quantify the performance of GPUs in term of how many billion floating-point operations it can perform per second). Acc1 and acc5 represents the proportion of the test samples for which the correct label is the model’s highest-confidence prediction of top-one and top-five accuracy.  



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
![Cosine Equation.](/home/adasi/git-sim/cosine_eq.png). And returns the calculated cosine similarity value.

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

<details>  
<summary> GFLOPS Range (0)</summary>
  
|**NN Architecture**|alexnet|efficientnet_b0|efficientnet_b1(v2)|mnasnet0_5|mnasnet0_75|mnasnet1_0|mnasnet1_3|mobilenet_v2 (v2)|mobilenet_v3_large(v2)|mobilenet_v3_small|regnet_x_400mf (v2)|
|-|-|-|-|-|-|-|-|-|-|-|-|
| **Positive Similarity**|-|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**| 22.9452|76.0708|106.3330|53.2820|55.9785|58.9318|64.6741|63.5908|62.2003|57.5352|74.1813|
| **Similarity Inference Time(ms)***|1.0910|1.9257|0.9677|1.7705|1.5173|1.9150|2.6190|1.6997|1.3955|1.5497|1.8296
| **Anchor & Positive Similarity**|54.0608|58.7266|62.6513|24.7069|34.0333|41.4679|30.7476|53.7780|81.3388|79.1964|85.3203|
| **Negative Similarity (n)**| -|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|66.6418|123.0218|135.6370|91.1896|92.5117|92.3293|92.2983|91.9986|102.9584|90.6320|112.9642|
| **Similarity Inference Time**| 1.7035|0.9840|1.3869|1.9929|1.0848|1.1435|1.0560|0.8957|0.9561|1.1070|1.0641|
| **Anchor & Negative Similarity**| 20.9051|-1.8293|-3.1438|1.4024|4.1157|14.5433|4.1319|25.2240|40.3210|35.2345|46.9519|
| **Negative Similarity (n1)-resized**| -|-|-|-|-|-|-|-|-|-|-|
| **Embedding Inference Time(ms)**|23.1783|85.9363|104.8846|55.4149|56.0505|55.1937|53.7698|61.7499|67.3475|65.4585|78.7289|
| **Similarity Inference Time**| 1.1210|0.8719|0.8240|1.5688|1.6725|1.8065|1.9388|1.4744|1.0903|1.9948|1.3828|
| **Anchor & Negative Similarity**| 21.2423|0.6875|0.5179|1.2124|6.0947|14.5142|4.8405|26.2946|39.1919|30.6165|49.9285|
| **GFLOPS (Giga Floating Point Operations Per Second)**| 0.71|0.39|0.69|0.1|0.21|0.31|0.53|0.3|0.22|0.06|0.41|
| **acc1**| 56.522|77.692|79.838|67.734|71.18|73.456|76.506|72.154|75.274|67.668|74.864|
| **acc5**|79.066|93.532|94.934|87.49|90.496|91.51|93.522|90.822|92.566|87.402|92.322|


### **Analysis:** 
- **alexnet** has the smallest embedding inference time (positive similarity)
- **efficientnet_b0** & **efficientnet_b1 (v2)** has negative values in the anchor & negative similarity (negativr similarity) of **image (n)**
- **efficientnet_b1(v2)** has the smallest similarity inference time (negative similarity)
- **mobilenet_v3_large(v2)** & **regnet_x_400mf (v2)** models reach +80% in anchor & positive similarity (positive similarity), while **mobilenet_v3_small** almost 80%
  
</details>











