# IndependentStudy (CSE 400E)

```
Independent Study Advisor: Professor Ulugbek Kamilov

Ph.D. Candidate Advisor: Weijie Gan

Computational Imaging Group Website: https://cigroup.wustl.edu/
```


## `1. [Meeting on 8/29/2022]`

> Meeting Purpose: To solve unstable and unusual training and validation performance.

`Unstable Training Tendency by Deep Unfolding Network`
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

TensorBoard  	      |      MRI Image
:---------------: | :-------------:
<img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188336324-fcf66f6b-7b51-403f-a248-dcf0bf09c79b.PNG">  | <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188514259-a5665cf4-7a74-455a-8f4b-7e7805b5ed25.png">


```diff
+ Analysis: Different from the original work of DeCoLearn, this work uses 12 sensitivity maps to utilize as much information as possible we can.

+          Therefore, I believe that indirectly affected registration performance impact on DeCoLearn's performance negatively.
```

#### `1.2.2. [Experiment Result - DeCoLearn Only Training 2  - With Gradient Clipping (clip_value=0.5)]`

TensorBoard  	      |      MRI Image
:---------------: | :-------------:
<img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188336333-66e94836-fb03-457a-9599-5c99cce0d143.PNG">  | <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188514262-20c75b58-4ce2-45ff-90b3-0429d9d8eee3.png">


```diff
+ Analysis: I applied Gradient Clipping at the reconstruction module's parameters and registration module's parameters at the same time.

+           Following that the DeCoLearn can work decently on the MRI dataset with 12 sensitivity maps.
```

#### `1.2.3. [Experiment Result - Deep Unrolling Training with DU iteration k = 7 / With Gradient Clipping (clip_value=1)]`

TensorBoard  	      |      MRI Image
:---------------: | :-------------:
<img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188336895-83ba5b00-97db-47e6-8274-4b0b95b08f23.PNG">  | <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188514038-cad7d09e-0978-441a-8e46-703103c72e26.png">

```diff
+ Analysis: Since I increase the number of parameters by setting iteration variable as 7, it shows the overfitting tendency when clip_value is 1.

```

#### `1.2.4. [Experiment Result - Deep Unrolling Training with DU iteration k = 7 / With Gradient Clipping (clip_value=0.5)]`

TensorBoard  	      |      MRI Image
:---------------: | :-------------:
<img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188336896-7e78305f-92b8-4406-bb42-93997547cc50.PNG">  | <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188513958-73b04cba-c48d-4daa-9ad9-d4d93f8bfaa9.png">


```diff
+ Analysis: By reducing the clip_value at the same model with 1.2.3, it presents stable training tendency.

```

#### `1.2.5. [Experiment Result - Deep Unrolling Training with DU iteration k = 10 / With Gradient Clipping (clip_value=0.3)]`

TensorBoard  	      |      MRI Image
:---------------: | :-------------:
<img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188513334-da57b11d-962a-4058-99b3-0a8737339d3a.jpeg">  | <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188513909-44087e20-eacf-4354-8ddb-b1a3a29ca0fc.png">

#### `1.2.6. [Experiment Result - Deep Unrolling Training with DU iteration k = 3 / With Gradient Clipping (clip_value=0.3)] batch = 4 / n_resblock = 5`

TensorBoard  	      |      MRI Image
:---------------: | :-------------:
<img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188631517-f87f4931-b928-4297-b08f-4aa5ba2e93e2.PNG">  | <img width="450" alt="IMG" src="https://user-images.githubusercontent.com/73331241/188631528-3f66f55a-1ac2-4c49-8706-8678d902aa81.png">

```diff
+ Analysis: Since the DU iteration parameter k is small with 3 and resblock is shallow with 5 blocks - because of the batch size of 4, we have unstable training result.

+           I will try larger parameters through server GPU.
```

----


#### `Conclusion of Meeting on 8/29/2022`

When I add a sensitivity map to the DeCoLearn, I had never expected unstable training performance because of the more plentiful data.

However, I can analyze that the additional sensitivity map suggested a more challenging environment for the Registration Module.

This research note will help future training issues.




Possible Problem: Incorrect coding in mul_coil processing.





<!--

----

Possible reason of the problem: Utilizing the sensitivity map

Solution: Gradient Clipping with the maximum size of 0.5


----

Meeting Date: 8/29/2022

Meeting Purpose: Server PC Set-up


----
-->
