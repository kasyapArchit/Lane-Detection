# Lane Detection<br>
This project to detect lanes on a road (it works for straight lanes not the curved ones)<br>

Basic pipeline:<br>
    1. Convert BGR to some other model which helps better visualization.<br>
    2. Isolate yellow and white color(the lane markers).<br>
    3. Convert to grayscale.<br>
    4. Apply Gaussian blur to smoothen image.<br>
    5. Use Canny edge detection to detect edges.<br>
    6. Discard edges which are not of interest.<br>
    7. Apply Hugh Transform to find lanes.<br>
    8. Interpolate the lanes and highlight them.<br>


Images:<br>
    1. https://github.com/udacity/CarND-LaneLines-P1/tree/master/test_images<br>
    2. Google Images<br>
    
Refrences:<br>
    1. https://towardsdatascience.com/finding-lane-lines-on-the-road-30cf016a1165<br>
    2. https://github.com/naokishibuya/car-finding-lane-lines<br>
    3. https://github.com/kenshiro-o/CarND-LaneLines-P1<br>

