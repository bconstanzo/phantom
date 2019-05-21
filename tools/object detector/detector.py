import dlib
import glob
import cv2

PATH_TRAIN_XML = "C:/Users/Ayrton/PycharmProjects/detector/train_1/train.xml"
PATH_TEST = "C:/Users/Ayrton/PycharmProjects/detector/test_1"


def train(path_xml):
    # objeto contenedor de las opciones para la rutina train_simple_object_detector()
    # todas las opciones vienen con valores por defecto razonables
    # http://dlib.net/python/index.html#dlib.simple_object_detector_training_options

    options = dlib.simple_object_detector_training_options()

    options.C = 6  # parametro C de las SVM, valores grandes pueden llevar al overfitting
    options.add_left_right_image_flips = True   # para objetos simetricos como las caras
    options.be_verbose = True
    options.epsilon = 0.005  # epsilon de detencion, valores pequeños -> accurate training

    dlib.train_simple_object_detector(path_xml, "detector.svm", options)

    # utilizando la herramienta https://imglab.ml/

    print("")
    print("Training accuracy: ", dlib.test_simple_object_detector(path_xml, "detector.svm"))


def test(path):
    detector = dlib.simple_object_detector("detector.svm")
    win = dlib.image_window()
    win.set_image(detector)
    dlib.hit_enter_to_continue()

    for filename in glob.glob(f"{path}/*.jpg"):
        imagen = cv2.imread(filename)
        dets = detector(imagen)

        print("Numero de detecciones: ", len(dets))
        for k, d in enumerate(dets):
            print("Deteccion {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))

        win.clear_overlay()
        win.set_image(imagen)
        win.add_overlay(dets)
        dlib.hit_enter_to_continue()


def main():
    train(PATH_TRAIN_XML)
    test(PATH_TEST)


if __name__ == "__main__":
    main()
