# image-cropper
A tool that can help you cropping, resizing and compressing images in an optimized way.

## Dependencies:
- PIL
- opencv2

## Usage:
You can call it as a command line tool like:
```
python imaging.py -i test.jpg -w 90 -l 160 -m file -f
```
The result will be printed out:
```
Cropped image file successfully! Image path is data\20160717\12\86985052.jpg
```
Command line arguments are as follows:
```
optional arguments:
  -h, --help  show this help message and exit
  -i INPUT    Url or filename of source image
  -w WIDTH    Width of cropped image
  -l HEIGHT   Height of cropped image
  -t TYPE     File type of cropped image, default is jpg
  -q QUALITY  Quality of cropped image, 1 to 100
  -f          Enable face detection
  -m MODE     Imaging mode: url | file | qr
```
Amoung them: 
- if width or height is not given, image will remains to its original size.
- -m set a mode for image cropper, **url** calls for cropping remote image; **file** calls for cropping local image by its file name; **qr** means detecting a local image whether it contains a QR code, the result is like: `Image did (not) contain QR code`
- -f enables face detection, in which cropping will base on the coordination of the face appeared in images.

Or you can import it as a module, and call it from your python code:
```
import imaging

result = imaging.crop_image(url,
                            width=160,
                            height=90,
                            img_type='jpg',
                            quality=80,
                            face_detect=1)
```

## Test
Original image:

![test_origin](./test.jpg)

In case 1, disable face detection:
```
python imaging.py -i test.jpg -w 90 -l 160 -m file
```
The result image:

![test_origin](./test_r1.jpg)

In case 2, enable face detection:
```
python imaging.py -i test.jpg -w 90 -l 160 -m file -f
```
The result image:

![test_origin](./test_r2.jpg)
