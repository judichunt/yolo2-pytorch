import cv2
import numpy as np

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg

cv2.setNumThreads(0)

h5_fname = 'models/yolo-voc.weights.h5'
trained_model = cfg.trained_model
thresh = 0.5
im_path = 'frames'
# ---

net = Darknet19()
net_utils.load_net(trained_model, net)
net.cuda()
net.eval()
print('load model succ...')

t_det = Timer()
t_total = Timer()

def frame_tran(image, i):

    im_data=np.expand_dims(
        yolo_utils.preprocess_test((image, None, cfg.multi_scale_inp_size), 0)[0], 0)

    t_total.tic()
    im_data = net_utils.np_to_variable(
        im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
    t_det.tic()
    bbox_pred, iou_pred, prob_pred = net(im_data)
    det_time = t_det.toc()
    # to numpy
    bbox_pred = bbox_pred.data.cpu().numpy()
    iou_pred = iou_pred.data.cpu().numpy()
    prob_pred = prob_pred.data.cpu().numpy()

    # print bbox_pred.shape, iou_pred.shape, prob_pred.shape

    bboxes, scores, cls_inds = yolo_utils.postprocess(
        bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh)

    im2show = yolo_utils.draw_detection(image, bboxes, scores, cls_inds, cfg)


    total_time = t_total.toc()

    if i % 1 == 0:
        format_str = 'frame: %d, ' \
                     '(detection: %.1f Hz, %.1f ms) ' \
                     '(total: %.1f Hz, %.1f ms)'
        print((format_str % (
            i,
            1. / det_time, det_time * 1000,
            1. / total_time, total_time * 1000)))

        t_total.clear()
        t_det.clear()
    return im2show


def play_save_cap(input_dir, filename, fps=0):
    cap = cv2.VideoCapture(input_dir + filename)
    if fps==0:
        fps=int(cap.get(5))
    cap.set(cv2.CAP_PROP_FPS, fps)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('Obj_'+filename, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    i=0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:

            frame=frame_tran(frame, i)
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        i+=1

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

input_dir = "videos/"
filename = "HomeOffice.avi"
play_save_cap(input_dir, filename, 24)