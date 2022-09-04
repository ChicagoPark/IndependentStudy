# IndependentStudy (CSE 400E)
```
WashU Research Course

Independent Study Advisor: Professor Ulugbek Kamilov

Ph.D. Candidate Advisor: Weijie Gan
```


## `1. [Meeting on 8/29/2022]`

> Meeting Purpose: To solve unstable and unusual training and validation performance.

`Unstable Training Tendency using Deep Unfolding Network`
:---------------: 
 <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188335358-5290760d-1651-4deb-89e8-fc89b3e250ae.PNG">   

----

### `1.1. [List of Feedback] (8/29/2022)`

----
```diff
+ Suggestion 1: Before I calculate the PSNR, add the following code (e.g. x[GROUND_TRUTH==0]) to remove the gray area at the background.

+ Suggestion 2: Try to use DeCoLearn only for training. 

+ Suggestion 3: Try Gradient Clipping (Set the maximum norm as 0.5) since exploding gradient can be one of the causes of this problem. 

+ Suggestion 4: Increase batch size. Since large batch size can ensure relatively stable training.
```

### `1.2. [Experiment Result]`

> I basically added PSNR code from all following experiments.

```python
fixed_y_tran_recon[fixed_x == 0] = 0
```

#### `1.2.1. [Experiment Result - DeCoLearn Only Training 1 - With Gradient Clipping (clip_value=1)]`

<img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188336324-fcf66f6b-7b51-403f-a248-dcf0bf09c79b.PNG">   


```diff
+ Analysis: Different from the original work of DeCoLearn, this work uses 12 sensitivity maps to utilize as much information as possible we can.

+          Therefore, I believe that indirectly affected registration performance impact on DeCoLearn's performance negatively.
```

#### `1.2.2. [Experiment Result - DeCoLearn Only Training 2  - With Gradient Clipping (clip_value=0.5)]`

<img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188336333-66e94836-fb03-457a-9599-5c99cce0d143.PNG">

```diff
+ Analysis: I applied Gradient Clipping at the reconstruction module's parameters and registration module's parameters at the same time.

+           Following that the DeCoLearn can work decently on MRI dataset with 12 sensitivity maps.
```

#### `1.2.3. [Experiment Result - Try Gradient Clipping]`

```diff
+ Analysis 1: 

```

#### `1.2.4. [Experiment Result - ]`

```diff
+ Analysis 1: 

```


----




----

Possible reason of the problem: Utilizing the sensitivity map

Solution: Gradient Clipping with the maximum size of 0.5


----

Meeting Date: 8/29/2022

Meeting Purpose: Server PC Set-up


----
