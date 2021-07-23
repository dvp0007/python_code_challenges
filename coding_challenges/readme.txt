1) Ability to download and edit data and use existing code correctly

Please try out a multiscale algorithm from the domain of texture synthesis to see how well it works for microstructure data. 

The algorithm can be found here: https://github.com/anopara/multi-resolution-texture-synthesis 
As microstructures, you can take some images from this data set: https://materialsdata.nist.gov/handle/11256/940 
Pick 2 or 3 structures from the data set, choose as you like. You might have to crop some parts of the images away such that only the structure itself remains. You can also decrease the resolution to make the code run faster
Please use the code to make a gif like those on the GitHub page.


2) Ability to understand Python + TensorFlow code and debug it

2.1) task_2.py is a small script that trains a neural networt using TensorFlow. But we introduced some bugs that you need to fix. If the program works properly, it will reach an accuracy of over 99 percent already after 3 epochs.

2.2) Answer this question: In task_2.py, what is the purpose of the decorator @tf.function?

2.3) Test different learning rates and plot the obtained accuracy over the learning rate using matplotlib. Save the plot to png.

3) Ability to rewrite code in a more efficient way

task_3.py is a script which marks all fields in a spatial domain x that are pores. The function works, but is really slow, especially if the resolution of x increases. Please implement a faster version that always yields the same result as the slow version. Do not do any new imports for this.

4) Ability to rearrange code without breaking it

task_4.py is quite messy. If you clean it up, you will also find a way to accelerate it substantially. This does not require an understanding of the mathematical equations, it is enough to analyze the data flows. Please only edit the function test_many_vectors() .
