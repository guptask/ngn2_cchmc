#NGN2 Analysis for Mount Sinai Images

##Packages to install

###opencv: 
>Clone the opencv repo at https://github.com/Itseez/opencv. Follow the 
instructions for installation on Linux at http://opencv.org/. Add to 
~/.bashrc :
```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib
```
To test sample opencv code, compile using 
```c++
g++ < file_name > `pkg-config opencv --cflags --libs`
```

##Build and run image analysis package

+ Inside the project root directory, type **make** to build the project.
A binary called **analyze** will be created.

+ Command to run the software:
```c++
./analyze < image directory path with / at end >
```

+ Image directory path should have a **jpg** directory which contains the 
separate jpg images for the RGB layers.

+ **image_list.dat** has to be created inside the image directory path. This 
tracks the different images that are being processed and allows selective 
processing of one or more images.

##Result

+ Inside the image directory path, a directory called **result** gets created. 
This contains the raw, enhanced and analyzed images for each image.

+ The **computed_metrics.csv** contains the metrics results generated during 
the analysis.

