from pyspark import SparkConf, SparkContext
conf = SparkConf().setMaster("local[*]").setAppName("My App")
sc = SparkContext.getOrCreate(conf = conf)
from subprocess import call
import os
import io
import numpy
from PIL import Image
import enhance
import shutil

#Video file:
video = "/opt/video.mp4"
#partitions should be 3~4x the number of cores on the cluster
partitions = 18

def enhanceImg(image):
    key = image[0]
    image = Image.open(io.BytesIO(image[1]))
    image = numpy.asarray(image, dtype=numpy.uint8)
    image = image.reshape((1080,1920,3))
    enhancer = enhance.NeuralEnhancer(loader=False)
    out = enhancer.process(image)
    name = os.path.basename(key)
    out.save("/opt/Output/ProcessedFrames/" + name, "JPEG")

#Remove leftover files from previous attempt and recreate dirs
if os.path.exists("/opt/Output"):
    shutil.rmtree("/opt/Output")
os.makedirs("/opt/Output")
os.makedirs("/opt/Output/UnprocessedFrames")
os.makedirs("/opt/Output/ProcessedFrames")

#extract audio
command = ["ffmpeg", "-i", video, "/opt/Output/audio.mp3"]
call(command)

#Extract frames
#For simplicity, assume video is 24 fps
command = ["ffmpeg", "-i", video, "-r", "24/1", "-q:v", "1",
        "/opt/Output/UnprocessedFrames/output%03d.jpg"]
call(command)

# Convert image to byte type and map them to value
images = sc.binaryFiles("/opt/Output/UnprocessedFrames/", partitions)
imageToArray = (lambda rawdata: np.asarray(Image.open(io.StringIO(rawdata))))
#Keys for the images should be their filenames
images.values().map(imageToArray)
images.foreach(enhanceImg)

#Combine frames into video with sound
command = ["ffmpeg", "-framerate", "24", "-i",
        "/opt/Output/ProcessedFrames/output%03d.jpg", "-i", "/opt/Output/audio.mp3", "-c:v",
        "libx264", "-vf", "fps=24", "-pix_fmt", "yuv420p", "/opt/Output/out.mp4"]
call(command)
