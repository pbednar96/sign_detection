import os
import io
import pandas as pd
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple

# input training CSV file and training images directory
TRAIN_CSV_FILE_LOC = os.getcwd() + "/training_data/" + "train_labels.csv"
TRAIN_IMAGES_DIR = os.getcwd() + "/training_images"

# input test CSV file and test images directory
EVAL_CSV_FILE_LOC = os.getcwd() + "/training_data/" + "eval_labels.csv"
TEST_IMAGES_DIR = os.getcwd() + "/test_images"

# training and testing output .tfrecord files
TRAIN_TFRECORD_FILE_LOC = os.getcwd() + "/training_data/" + "train.tfrecord"
EVAL_TFRECORD_FILE_LOC = os.getcwd() + "/training_data/" + "eval.tfrecord"

#######################################################################################################################
def main():

    trainTfRecordFileWriteSuccessful = writeTfRecordFile(TRAIN_CSV_FILE_LOC, TRAIN_TFRECORD_FILE_LOC, TRAIN_IMAGES_DIR)
    if trainTfRecordFileWriteSuccessful:
        print("successfully created the training TFRecords, saved to: " + TRAIN_TFRECORD_FILE_LOC)

    evalTfRecordFileWriteSuccessful = writeTfRecordFile(EVAL_CSV_FILE_LOC, EVAL_TFRECORD_FILE_LOC, TEST_IMAGES_DIR)
    if evalTfRecordFileWriteSuccessful:
        print("successfully created the eval TFRecords, saved to: " + EVAL_TFRECORD_FILE_LOC)

#######################################################################################################################
def writeTfRecordFile(csvFileName, tfRecordFileName, imagesDir):
    # use pandas to read in the .csv file data, pandas.read_csv() returns a type DataFrame with the given param
    csvFileDataFrame = pd.read_csv(csvFileName)

    # reformat the CSV data into a format TensorFlow can work with
    csvFileDataList = reformatCsvFileData(csvFileDataFrame)

    # instantiate a TFRecordWriter for the file data
    tfRecordWriter = tf.python_io.TFRecordWriter(tfRecordFileName)

    # for each file (not each line) in the CSV file data . . .
    # (each image/.xml file pair can have more than one box, and therefore more than one line for that file in the CSV file)
    for singleFileData in csvFileDataList:
        tfExample = createTfExample(singleFileData, imagesDir)
        tfRecordWriter.write(tfExample.SerializeToString())
    # end for
    tfRecordWriter.close()
    return True        # return True to indicate success
# end function

#######################################################################################################################
def reformatCsvFileData(csvFileDataFrame):
    # the purpose of this function is to translate the data from one CSV file in pandas.DataFrame format
    # into a list of the named tuple below, which then can be fed into TensorFlow

    # establish the named tuple data format
    dataFormat = namedtuple('data', ['filename', 'object'])

    #  pandas.DataFrame.groupby() returns type pandas.core.groupby.DataFrameGroupBy
    csvFileDataFrameGroupBy = csvFileDataFrame.groupby('filename')

    # declare, populate, and return the list of named tuples of CSV data
    csvFileDataList = []
    for filename, x in zip(csvFileDataFrameGroupBy.groups.keys(), csvFileDataFrameGroupBy.groups):
        csvFileDataList.append(dataFormat(filename, csvFileDataFrameGroupBy.get_group(x)))
    # end for
    return csvFileDataList
# end function

#######################################################################################################################
def createTfExample(singleFileData, path):
    # use TensorFlow's GFile function to open the .jpg image matching the current box data
    with tf.gfile.GFile(os.path.join(path, '{}'.format(singleFileData.filename)), 'rb') as tensorFlowImageFile:
        tensorFlowImage = tensorFlowImageFile.read()
    # end with

    # get the image width and height via converting from a TensorFlow image to an io library BytesIO image,
    # then to a PIL Image, then breaking out the width and height
    bytesIoImage = io.BytesIO(tensorFlowImage)
    pilImage = Image.open(bytesIoImage)
    width, height = pilImage.size

    # get the file name from the file data passed in, and set the image format to .jpg
    fileName = singleFileData.filename.encode('utf8')
    imageFormat = b'jpg'

    # declare empty lists for the box x, y, mins and maxes, and the class as text and as an integer
    xMins = []
    xMaxs = []
    yMins = []
    yMaxs = []
    classesAsText = []
    classesAsInts = []

    # for each row in the current .xml file's data . . . (each row in the .xml file corresponds to one box)
    for index, row in singleFileData.object.iterrows():
        xMins.append(row['xmin'] / width)
        xMaxs.append(row['xmax'] / width)
        yMins.append(row['ymin'] / height)
        yMaxs.append(row['ymax'] / height)
        classesAsText.append(row['class'].encode('utf8'))
        classesAsInts.append(classAsTextToClassAsInt(row['class']))
    # end for

    # finally we can calculate and return the TensorFlow Example
    tfExample = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(fileName),
        'image/source_id': dataset_util.bytes_feature(fileName),
        'image/encoded': dataset_util.bytes_feature(tensorFlowImage),
        'image/format': dataset_util.bytes_feature(imageFormat),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xMins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xMaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(yMins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(yMaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classesAsText),
        'image/object/class/label': dataset_util.int64_list_feature(classesAsInts)}))

    return tfExample
# end function

#######################################################################################################################
def classAsTextToClassAsInt(classAsText):

    if classAsText == 'dej_prednost_v_jizde':
        return 1
    elif classAsText == 'zakaz_vjezdu_vsech_vozidel':
        return 2
    elif classAsText == 'zakaz_vjezdu_v_obou_smerech':
        return 3
    elif classAsText == 'stop':
        return 4
    elif classAsText == 'zakaz_odboceni_vlevo':
        return 5
    elif classAsText == 'zakaz_odboceni_vpravo':
        return 6
    elif classAsText == 'zakaz_stani':
        return 7
    elif classAsText == 'zakaz_zastaveni':
        return 8
    elif classAsText == 'hlavni_pozemni_komunikace':
        return 9
    elif classAsText == 'konec_hlavni_pozemni_komunikace':
        return 10
    elif classAsText == 'krizovatka_s_vedlejsi_pozemni_komunikaci':
        return 11
    elif classAsText == 'pozor_zpomalovaci_prah':
        return 12
    elif classAsText == 'parkoviste':
        return 13
    elif classAsText == 'slepa_ulice':
        return 14
    elif classAsText == 'jednosmerny_provoz':
        return 15
    elif classAsText == 'prechod_pro_chodce':
        return 16
    elif classAsText == 'kruhovy_objezd':
        return 17
    elif classAsText == '30':
        return 18
    elif classAsText == '40':
        return 19
    elif classAsText == '50':
        return 20
    elif classAsText == '60':
        return 21
    elif classAsText == '70':
        return 22
    elif classAsText == '80':
        return 23
    elif classAsText == 'prikazany_smer_jizdy_vlevo':
        return 24
    elif classAsText == 'prikazany_smer_jizdy_vpravo':
        return 25
    else:
        print("error in class_text_to_int(), row_label could not be identified")
        return -1
    # end if
# end function

#######################################################################################################################
if __name__ == '__main__':
    main()